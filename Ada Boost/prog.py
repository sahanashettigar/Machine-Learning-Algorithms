import numpy as np
from numpy.random import sample
from sklearn.tree import DecisionTreeClassifier

"""
Use DecisionTreeClassifier to represent a stump.
------------------------------------------------
DecisionTreeClassifier Params:
    critereon -> entropy
    max_depth -> 1
    max_leaf_nodes -> 2
Use the same parameters
"""
# REFER THE INSTRUCTION PDF FOR THE FORMULA TO BE USED


class AdaBoost:

    """
    AdaBoost Model Class
    Args:
        n_stumps: Number of stumps (int.)
    """

    def __init__(self, n_stumps=20):

        self.n_stumps = n_stumps
        self.stumps = []

    def fit(self, X, y):
        """
        Fitting the adaboost model
        Args:
            X: M x D Matrix(M data points with D attributes each)(numpy float)
            y: M Vector(Class target for all the data points as int.)
        Returns:
            the object itself
        """
        self.alphas = []

        sample_weights = np.ones_like(y) / len(y)
        for _ in range(self.n_stumps):

            st = DecisionTreeClassifier(
                criterion='entropy', max_depth=1, max_leaf_nodes=2)
            st.fit(X, y, sample_weights)
            y_pred = st.predict(X)

            self.stumps.append(st)

            error = self.stump_error(y, y_pred, sample_weights=sample_weights)
            alpha = self.compute_alpha(error)
            self.alphas.append(alpha)
            sample_weights = self.update_weights(
                y, y_pred, sample_weights, alpha)
        return self

    def stump_error(self, y, y_pred, sample_weights):
        """
        Calculating the stump error
        Args:
            y: M Vector(Class target for all the data points as int.)
            y_pred: M Vector(Class target predicted for all the data points as int.)
            sample_weights: M Vector(Weight of each sample float.)
        Returns:
            The error in the stump(float.)
        """
        errorarr = [0 for i in range(len(y))]
        for i in range(0, len(y)):
            if(y[i] != y_pred[i]):
                errorarr[i] = 1
        errorarr = np.array(errorarr)
        error = np.sum((np.multiply(errorarr, sample_weights)))
        return error

    def compute_alpha(self, error):
        """
        Computing alpha
        The weight the stump has in the final prediction
        Use eps = 1e-9 for numerical stabilty.
        Args:
            error:The stump error(float.)
        Returns:
            The alpha value(float.)
        """
        eps = 1e-9
        alpha = 0.5*np.log((1-error)/(error+eps))
        return alpha

    def update_weights(self, y, y_pred, sample_weights, alpha):
        """
        Updating Weights of the samples based on error of current stump
        The weight returned is normalized
        Args:
            y: M Vector(Class target for all the data points as int.)
            y_pred: M Vector(Class target predicted for all the data points as int.)
            sample_weights: M Vector(Weight of each sample float.)
            alpha: The stump weight(float.)
        Returns:
            new_sample_weights:  M Vector(new Weight of each sample float.)
        """
        new_weight = [0 for i in range(len(sample_weights))]
        for i in range(len(sample_weights)):
            if y[i] != y_pred[i]:
                new_weight[i] = (sample_weights[i] /
                                 float(self.n_stumps))*np.exp(alpha)
            else:
                new_weight[i] = (sample_weights[i] /
                                 float(self.n_stumps))*np.exp(-alpha)
        normalised_weights = [(i/sum(new_weight)) for i in new_weight]
        return normalised_weights

    def predict(self, X):
        """
        Predicting using AdaBoost model with all the decision stumps.
        Decison stump predictions are weighted.
        Args:
            X: N x D Matrix(N data points with D attributes each)(numpy float)
        Returns:
            pred: N Vector(Class target predicted for all the inputs as int.)
        """
        output = [{} for i in range(X.shape[0])]
        for i in range(len(self.stumps)):
            classifier = self.stumps[i]
            y = classifier.predict(X)
            for j in range(len(y)):
                if y[j] in output[j].keys():
                    output[j][y[j]] += self.alphas[i]
                else:
                    output[j][y[j]] = self.alphas[i]
        pred = [0 for i in range(X.shape[0])]
        for i in range(len(output)):
            pred[i] = max(output[i], key=output[i].get)
        return pred

    def evaluate(self, X, y):
        """
        Evaluate Model on test data using
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix
            y: True target of test data
        Returns:
            accuracy : (float.)
        """
        pred = self.predict(X)
        # find correct predictions
        correct = (pred == y)

        accuracy = np.mean(correct) * 100  # accuracy calculation
        return accuracy
