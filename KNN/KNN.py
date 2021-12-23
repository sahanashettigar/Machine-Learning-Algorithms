import numpy as np


class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data
        self.target = target.astype(np.int64)

        return self

    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        if(len(x) == 0):
            return []
        sol = [[0 for i in range(len(self.data))] for j in range(len(x))]
        for i in range(len(x)):
            for j in range(len(self.data)):
                sol[i][j] = self.minkowski(x[i], self.data[j])
        return sol

    def minkowski(self, test_data, train_data):
        minkowski_dist = 0
        for i in range(len(test_data)):
            minkowski_dist += np.power(abs(test_data[i]-train_data[i]), self.p)
        if(minkowski_dist > 0):
            minkowski_dist = np.power(minkowski_dist, float(1/self.p))
        return minkowski_dist

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        if(len(x) == 0):
            return []
        distance = self.find_distance(x)
        neigh_dists = []
        idx_of_neigh = []
        for i in range(len(distance)):
            idx_output = np.argsort(distance[i])
            idx_of_neigh.append(idx_output[:self.k_neigh])
            dist_output = sorted(distance[i])[:self.k_neigh]
            neigh_dists.append(dist_output)
        return [neigh_dists, idx_of_neigh]

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        if(len(x) == 0):
            return []
        solutionarr = self.k_neighbours(x)
        if (len(self.target) == 0):
            return 0
        id_of_neighs = solutionarr[1]
        neigh_dists = solutionarr[0]
        sol = []
        for id in range(len(id_of_neighs)):
            class_count = {}
            for i in range(len(id_of_neighs[id])):
                index_target = id_of_neighs[id][i]
                if(self.weighted == False):
                    if(self.target[index_target] in class_count.keys()):
                        class_count[self.target[index_target]] += 1
                    else:
                        class_count[self.target[index_target]] = 1
                elif(self.weighted == True):
                    if(self.target[index_target] in class_count.keys()):
                        class_count[self.target[index_target]
                                    ] += (1/(neigh_dists[id][i]+0.000000001))
                    else:
                        class_count[self.target[index_target]
                                    ] = (1/(neigh_dists[id][i]+0.000000001))
            class_count = dict(
                sorted(class_count.items(), key=lambda x: x[0], reverse=False))
            sol.append(max(class_count, key=class_count.get))
        return sol

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        output_x = self.predict(x)
        if (len(output_x) != len(y)):
            return 0.0
        true_count = 0
        for i in range(len(output_x)):
            if output_x[i] == y[i]:
                true_count += 1
        return float(true_count/len(output_x))*100
