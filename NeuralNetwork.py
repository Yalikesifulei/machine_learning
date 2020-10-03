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

    def predict(self, x):
        """
        Compute output for input vector x using forward propagation.
        
        Returned value is vector in shape of correct output.
        """
        a = x.copy()
        for j in range(1, self.LayersCount):
            a = sigmoid(self.Theta[j-1] @ np.append(1, a))
        return a

    def Cost(self, X, y, Regularization = 0):
        """
        Compute cost function for inputs matrix X and correct outputs y.
        X and y must be numpy arrays w/ shapes (m, n) and (m, 1) respectively.

        Regularization (float)     - regularization parameter.

        Returned value is float.
        """
        m, K = len(y), self.Layers[-1]
        J = -1/m * sum([
            sum([
                y[i][k]*(np.log(self.predict(X[i]))[k]) + (1-y[i][k])*(np.log(1-self.predict(X[i]))[k])
                for k in range(K)
            ]) for i in range(m) 
        ])
        T = self.Theta.copy()
        for t in T:
            t[:,0] = 0
        J += Regularization/(2*m) * sum([np.linalg.norm(t, ord = "fro")**2 for t in T])
        return J

    def Grad(self, X, y, Regularization = 0):
        """
        Compute gradient of cost function for inputs matrix X and correct outputs y
        using back propagation.
        X and y must be numpy arrays w/ shapes (m, n) and (m, 1) respectively.

        Regularization (float)     - regularization parameter.

        Returned value is in vector form.
        """
        m = len(y)
        Delta, a = np.empty(self.LayersCount-1, dtype=np.ndarray), []
        DVec = np.array([])
        for l in range(self.LayersCount-1):
            Delta[l] = np.zeros_like(self.Theta[l])
        for l in self.Layers:
            a.append(np.zeros((l+1, m)))
        a[-1] = np.zeros((self.Layers[-1], m))
        a[0] = np.vstack((np.ones((1, m)), X.T))
        a = np.array(a)
        delta = np.zeros_like(a)

        # forward propagation
        for j in range(1, self.LayersCount-1):
            a[j] = np.vstack((np.ones((1, m)), sigmoid(self.Theta[j-1] @  a[j-1])))
        a[-1] = sigmoid(self.Theta[-1] @ a[-2])

        # backward propagation
        delta[-1] = a[-1] - y.T
        for j in range(self.LayersCount-2, 0, -1):
            delta[j] = ((self.Theta[j].T @ delta[j+1]) * (a[j] * (1-a[j])))[1:, ]
        for l in range(self.LayersCount-1):
            Delta[l] += delta[l+1] @ a[l].T
        T = self.Theta.copy()
        for t in T:
            t[:,0] = 0
        Delta = (1/m) * Delta + Regularization*T
        for d in Delta:
            DVec = np.append(DVec, d)
        return DVec

    def NumGrad(self, X, y, Regularization = 0):
        """
        Compute gradient of cost function for inputs matrix X and correct outputs y
        using finite differences.
        X and y must be numpy arrays w/ shapes (m, n) and (m, 1) respectively.

        Regularization (float)     - regularization parameter.

        Returned value is in vector form.
        Recommended to use only for debugging purpose because of slow computations.
        """
        DVec = np.zeros_like(self.ThetaVec)
        eps = 10e-5
        for i in range(len(DVec)):
            self.ThetaVec[i] += eps
            self.rollTheta()
            jPlus = self.Cost(X, y, Regularization)
            self.ThetaVec[i] -= 2*eps
            self.rollTheta()
            jMinus = self.Cost(X, y, Regularization)
            self.ThetaVec[i] += eps
            self.rollTheta()
            DVec[i] = (jPlus - jMinus)/(2*eps)
        return DVec

    def GradDesc(self, X, y, LearningRate = 0.01, Regularization = 0, Tolerance = 1e-4, MaxIter = 400, PrintStep = 1):
        """
        Perform gradient descent to fit weights for inputs matrix X and correct outputs y.
        X and y must be numpy arrays w/ shapes (m, n) and (m, 1) respectively.

        LearningRate (float)        - learning rate.
        Regularization (float)      - regularization constant.
        Tolerance (float)           - algorithm stops, when two Cost values differ less.
        MaxIter (int)               - maximum number of iterations.
        PrintStep (int)             - step for printing iteration info.

        Returns array of values of cost functions at each iteration.
        """
        J_hist = []
        # perform gradient descent, but Theta's are unrolled into vector
        for i in range(MaxIter):
            self.ThetaVec -= LearningRate*self.Grad(X, y, Regularization)
            self.rollTheta()
            if i%PrintStep == 0:
                print(f"iteration {i} \t J = {self.Cost(X, y, Regularization)}")
            J_hist.append(self.Cost(X, y, Regularization))
            if i>0:
                if abs(J_hist[i] - J_hist[i-1]) < Tolerance:
                    break
        print("--- GradDesc finished ---")
        return J_hist

    def SGD(self, X, y, BatchSize = 1, LearningRate = 0.01, Tolerance = 1e-4, MaxIter = 400, PrintStep = 1):
        """
        Perform stochastic gradients descent to fit weights for inputs matrix X and correct outputs y.
        X and y must be numpy arrays w/ shapes (m, n) and (m, 1) respectively.

        LearningRate (float)        - learning rate.
        BatchSize (float)           - size of mini-batch used to compute gradient.
        Tolerance (float)           - algorithm stops, when two Cost values differ less.
        MaxIter (int)               - maximum number of iterations.
        PrintStep (int)             - step for printing iteration info.

        Returns array of values of cost functions at each iteration.
        """
        J_hist = []
        m = len(y)
        X_shuffle = X.copy()
        for i in range(MaxIter):
            np.random.shuffle(X_shuffle)
            for j in range(0, m, BatchSize):
                self.ThetaVec -= LearningRate*self.Grad(X[j:j+BatchSize], y[j:j+BatchSize], Regularization = 0)
                self.rollTheta()
            if i%PrintStep == 0:
                print(f"iteration {i} \t J = {self.Cost(X, y, Regularization = 0)}")
            J_hist.append(self.Cost(X, y, Regularization = 0))
            if i>0:
                if abs(J_hist[i] - J_hist[i-1]) < Tolerance:
                    break
        print("--- SGD finished ---")
        return J_hist