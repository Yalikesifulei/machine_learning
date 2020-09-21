import numpy as np

def sigmoid(z):
    return (1+np.exp(-z))**(-1)

class Network:
    def __init__(self, Layers):
        """ 
        Layers - tupple or list with numbers of neurons in each layer.
        Initializes neural network of given size with random weights.
        """
        self.LayersCount = len(Layers)
        self.Layers = Layers
        self.Theta = np.empty(self.LayersCount-1, dtype=np.ndarray)
        self.ThetaVec = np.array([])
        for j in range(self.LayersCount-1):
            eps_init = np.sqrt(6)/np.sqrt(self.Layers[j] + self.Layers[j+1])
            self.Theta[j] = np.random.rand(Layers[j+1], Layers[j]+1)*2*eps_init - eps_init
            self.ThetaVec = np.append(self.ThetaVec, self.Theta[j].flatten())

    def rollTheta(self):
        """
        Update weight matrices using weight vector.
        Used since all related computations are performed in vector form.
        """
        ind_start, ind_end = 0, 0
        size_1, size_2 = 0, 0
        for j in range(self.LayersCount-1):
            size_1 = np.shape(self.Theta[j])[0]
            size_2 = np.shape(self.Theta[j])[1]
            ind_end += size_1 * size_2
            RedVec = self.ThetaVec[ind_start:ind_end]
            RedVec = RedVec.reshape(len(RedVec), 1)
            self.Theta[j] = RedVec.reshape(size_1, size_2)
            ind_start += (ind_end - ind_start)
            ind_end += (ind_end - ind_start)

    def h(self, x):
        """
        Compute output for input vector x using forward propagation.
        
        Returned value is vector in shape of correct output.
        """
        a = x.copy()
        for j in range(1, self.LayersCount):
            a = sigmoid(self.Theta[j-1] @ np.append(1, a))
        return a

    def Cost(self, X, y, lmb = 0):
        """
        Compute cost function for inputs matrix X and correct outputs y.
        X and y must be numpy arrays w/ shapes (m, n) and (m, 1) respectively.

        lmb (float)     - regularization parameter.

        Returned value is float.
        """
        m, K = len(y), self.Layers[-1]
        J = -1/m * sum([
            sum([
                y[i][k]*(np.log(self.h(X[i]))[k]) + (1-y[i][k])*(np.log(1-self.h(X[i]))[k])
                for k in range(K)
            ]) for i in range(m) 
        ])
        T = self.Theta.copy()
        for t in T:
            t[:,0] = 0
        J += lmb/(2*m) * sum([np.linalg.norm(t, ord = "fro")**2 for t in T])
        return J

    def Grad(self, X, y, lmb = 0):
        """
        Compute gradient of cost function for inputs matrix X and correct outputs y
        using finite differences.
        X and y must be numpy arrays w/ shapes (m, n) and (m, 1) respectively.

        lmb (float)     - regularization parameter.

        Returned value is in vector form.
        """
        m = len(y)
        Delta, D, a, delta = [], [], [], [0]*self.LayersCount
        DVec = np.array([])
        for l in range(self.LayersCount-1):
            Delta.append(np.zeros_like(self.Theta[l]))
            D.append(np.zeros_like(self.Theta[l]))
        Delta = np.array(Delta)
        D = np.array(D)
        for l in self.Layers:
            a.append(np.zeros(l+1))
        a[-1] = np.zeros(self.Layers[-1])
        a = np.array(a)

        for i in range(m):
            a[0] = np.append(1, X[i])
            # forward propagation
            for j in range(1, self.LayersCount-1):
                a[j] = np.append(1, sigmoid(self.Theta[j-1] @  a[j-1]))
            a[-1] = sigmoid(self.Theta[-1] @ a[-2])
            # compute delta errors
            delta[-1] = a[-1] - y[i]
            for j in range(self.LayersCount-2, 0, -1):
                delta[j] = ((self.Theta[j].T @ delta[j+1]) * (a[j] * (1-a[j])))[1:]
            for l in range(self.LayersCount-1):
                Delta[l] += delta[l+1].reshape(len(delta[l+1]), 1) @ a[l].reshape(len(a[l]), 1).T
        D = Delta.copy()
        T = self.Theta.copy()
        for t in T:
            t[:,0] = 0
        D = (1/m) * D + lmb*T
        for d in D:
            DVec = np.append(DVec, d)
        return DVec

    def NumGrad(self, X, y, lmb = 0):
        """
        Compute gradient of cost function for inputs matrix X and correct outputs y
        using finite differences.
        X and y must be numpy arrays w/ shapes (m, n) and (m, 1) respectively.

        lmb (float)     - regularization parameter.

        Returned value is in vector form.
        Recommended to use only for debugging purpose because of slow computations.
        """
        DVec = np.zeros_like(self.ThetaVec)
        eps = 10e-5
        for i in range(len(DVec)):
            self.ThetaVec[i] += eps
            self.rollTheta()
            jPlus = self.Cost(X, y, lmb)
            self.ThetaVec[i] -= 2*eps
            self.rollTheta()
            jMinus = self.Cost(X, y, lmb)
            self.ThetaVec[i] += eps
            self.rollTheta()
            DVec[i] = (jPlus - jMinus)/(2*eps)
        return DVec

    def GradDesc(self, X, y, alpha = 0.01, lmb = 0, eps = 1e-4, MaxIter = 400):
        """
        Perform gradient descent to fit weights for inputs matrix X and correct outputs y.
        X and y must be numpy arrays w/ shapes (m, n) and (m, 1) respectively.

        alpha (float)   - learning rate.
        lmb (float)     - regularization parameter.
        eps (float)     - stopping threshold.
        MaxIter (int)   - maximum number of iterations.

        Returns array of values of cost functions at each iteration.
        """
        J_hist = []
        # perform gradient descent, but Theta's are unrolled into vector
        for i in range(MaxIter):
            self.ThetaVec -= alpha*self.NumGrad(X, y, lmb)
            self.rollTheta()
            print(f"iteration {i+1} \t J = {self.Cost(X, y, lmb)}")
            J_hist.append(self.Cost(X, y, lmb))
            if i>0:
                if abs(J_hist[i] - J_hist[i-1]) < eps:
                    break
        print("--- GradDesc finished ---")
        return J_hist

    def SGD(self, X, y, BatchSize, alpha = 0.01, eps = 1e-4, MaxIter = 400):
        """
        Perform stochastic gradients descent to fit weights for inputs matrix X and correct outputs y.
        X and y must be numpy arrays w/ shapes (m, n) and (m, 1) respectively.

        BatchSize (int) - size of data batch used to compute gradient.
        alpha (float)   - learning rate.
        eps (float)     - stopping threshold.
        MaxIter (int)   - maximum number of iterations.

        Returns array of values of cost functions at each iteration.
        """
        J_hist = []
        m = len(y)
        X_shuffle = X.copy()
        for i in range(MaxIter):
            np.random.shuffle(X_shuffle)
            for j in range(0, m, BatchSize):
                self.ThetaVec -= alpha*self.Grad(X[j:j+BatchSize], y[j:j+BatchSize], lmb = 0)
                self.rollTheta()
            print(f"iteration {i+1} \t J = {self.Cost(X, y, lmb = 0)}")
            J_hist.append(self.Cost(X, y, lmb = 0))
            if i>0:
                if abs(J_hist[i] - J_hist[i-1]) < eps:
                    break
        print("--- SGD finished ---")
        return J_hist