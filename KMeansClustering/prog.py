import numpy as np


class KMeansClustering:
    """
    K-Means Clustering Model

    Args:
        n_clusters: Number of clusters(int)
    """

    def __init__(self, n_clusters, n_init=10, max_iter=1000, delta=0.001):

        self.n_cluster = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.delta = delta

    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])

    def fit(self, data):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix(M data points with D attributes each)(numpy float)
        Returns:
            The object itself
        """
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is grater than number of datapoints')

        best_centroids = None
        m_score = float('inf')

        for _ in range(self.n_init):
            self.init_centroids(data)

            for _ in range(self.max_iter):
                cluster_assign = self.e_step(data)
                old_centroid = np.copy(self.centroids)
                self.m_step(data, cluster_assign)

                if np.abs(old_centroid - self.centroids).sum() < self.delta:
                    break

            cur_score = self.evaluate(data, cluster_assign)

            if cur_score < m_score:
                m_score = cur_score
                best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids

        return self

    def e_step(self, data):
        """
        Expectation Step.
        Finding the cluster assignments of all the points in the data passed
        based on the current centroids
        Args:
            data: M x D Matrix (M training samples with D attributes each)(numpy float)
        Returns:
            Cluster assignment of all the samples in the training data
            (M) Vector (M number of samples in the train dataset)(numpy int)
        """
        cluster_group = np.zeros(data.shape[0], dtype=np.int)
        distance = np.zeros((self.n_cluster, data.shape[0]))
        for centroid in range(len(self.centroids)):
            for data_point in range(len(data)):
                dist = np.linalg.norm(
                    self.centroids[centroid]-data[data_point])
                distance[centroid][data_point] = dist
        dist_trans = distance.transpose()
        for cluster_decision in range(len(dist_trans)):
            minval = np.amin(dist_trans[cluster_decision])
            result = np.where(dist_trans[cluster_decision] == minval)
            cluster_group[cluster_decision] = result[0]
        return cluster_group

    def m_step(self, data, cluster_assgn):
        """
        Maximization Step.
        Compute the centroids
        Args:
            data: M x D Matrix(M training samples with D attributes each)(numpy float)
            cluster_assign: Cluster Assignment
        Change self.centroids
        """
        new_centroids = np.zeros((self.n_cluster, data.shape[1]))
        array = [0 for i in range(self.n_cluster)]
        for i in range(len(cluster_assgn)):
            new_centroids[cluster_assgn[i]] = np.add(
                new_centroids[cluster_assgn[i]], data[i])
            array[cluster_assgn[i]] += 1
        for i in range(len(new_centroids)):
            new_centroids[i] = np.true_divide(new_centroids[i], array[i])
        self.centroids = new_centroids

    def evaluate(self, data, cluster_assign):
        """
        K-Means Objective
        Args:
            data: Test data (M x D) matrix (numpy float)
            cluster_assign: M vector, Cluster assignment of all the samples in `data`
        Returns:
            metric : (float.)
        """
        metric = 0
        for i in range(len(data)):
            sum_for_ele = np.linalg.norm(
                data[i]-self.centroids[cluster_assign[i]])**2
            metric = metric + (sum_for_ele)
        return metric
