import numpy as np
import sys
from copy import deepcopy
from scipy.io import arff

from scipy.spatial import KDTree
from scipy.spatial import distance


class Cluster:
    def __init__(self, shape, point=None):

        if (point is not None):
            self.points = np.matrix(point)
            self.center = point
            self.rep = np.matrix(point)

        else:
            self.points = np.empty(shape=(0, shape[1]))
            self.center = None
            self.rep = np.empty(shape=(0, shape[1]))

        self.closest = None
        self.distance_closest = float('inf')


class Cure:
    def __init__(self, data, number_of_clusters, alpha, c):

        # data is a Numpy-array with rows as points and k
        # is the number of clusters
        self.data = data
        self.k = number_of_clusters
        self.alpha = alpha
        self.c = c
        # Stores representatives for each cluster
        self.KDTree = KDTree(data)

        self.shape = data.shape

        # Initializes each point as a Cluster object
        data_as_clusters = [Cluster(self.shape, point) for point in data]

        # Initializes each Clusters closest Cluster and distance using the
        # KDTree
        for cluster in data_as_clusters:
            query = self.KDTree.query(cluster.points[0], 2)
            #print query
            cluster.distance_closest = query[0][0][1]
            cluster.closest = data_as_clusters[query[1][0][1]]

        # Stores an entry for each cluster sorted by their distances to their
        # closest cluster
        self.Heap = sorted(data_as_clusters, key=lambda x:
        x.distance_closest, reverse=False)

    def cure_clustering(self):
        while (len(self.Heap) > self.k):

            # Select an arbitrary cluster from heap
            cluster_u = self.Heap[0]
            cluster_v = cluster_u.closest

            # remove to be merged elements from the heap and resort heap
            self.Heap.remove(cluster_v)
            self.Heap.remove(cluster_u)

            # merge the clusters to form a new cluster
            cluster_w = self.merge_cluster(cluster_u, cluster_v)

            # TODO: delete rep entries from kdtree and add w representative
            tree_data = np.empty(shape=(0, self.shape[1]))

            for cluster in self.Heap:
                for rep in cluster.rep:
                    tree_data = np.concatenate((tree_data, rep))

            for rep in cluster_w.rep:
                 tree_data = np.concatenate((tree_data, rep))

            print tree_data

            self.KDTree = KDTree(np.matrix(tree_data))

            # select arbitrary element from the heap
            cluster_w.closest = self.Heap[0]
            cluster_w.distance_closest = self.distance_func(cluster_w.center,
                                                            cluster_w.closest.center)

            for cluster in self.Heap:

                dist = self.distance_func(cluster_w.center, cluster.center)

                if (dist < cluster_w.distance_closest):
                    cluster_w.closest = cluster
                    cluster_w.distance_closest = dist

                if ((cluster.closest is cluster_u) or (cluster.closest is cluster_v)):

                    if (cluster.distance_closest < dist):
                        # get closest element to cluster with maximum distance
                        # dist TODO: check this
                        (cluster.distance_closest, cluster.closest) = self.closest_cluster(cluster, cluster_w, dist)

                    if (cluster.closest is None):
                        cluster.closest = cluster_w
                        cluster.distance_closest = dist

                    else:
                        cluster.closest = cluster_w
                        cluster.distance_closest = distance

                elif (cluster.distance_closest > dist):
                    cluster.closest = cluster_w
                    cluster.distance_closest = dist

            self.Heap.append(cluster_w)
            self.Heap.sort(key=lambda x: x.distance_closest, reverse=False)

        return 0

    def merge_cluster(self, cluster1, cluster2):
        # TODO: add merge function from paper
        merged_cluster = self.union_func(cluster1, cluster2)
        merged_cluster.center = (len(cluster1.points) * cluster1.center + len(cluster2.points) * cluster2.center) / (
        len(cluster1.points) + len(cluster2.points))
        tmpSet = []
        merged_cluster.rep = np.empty(shape=(0, self.shape[1]))
        for i in range(0, self.c):
            maxDist = 0
            maxPoint = []
            for point in merged_cluster.points:
                if i == 0:
                    minDist = self.distance_func(point, merged_cluster.center)

                else:
                    tmpDist = min([self.distance_func(point, p) for p in tmpSet])

                if minDist >= maxDist:
                    maxDist = minDist
                    maxPoint = point

            if not any((maxPoint == x).all() for x in tmpSet):
            	tmpSet.append(maxPoint)

        for i in xrange(0, len(tmpSet)):
            # Smaller alpha shrinks the scattered points and favors elongated clusters
            # large alph-> scattered points get closer to mean,  cluster tend to be more compact
            # merged_cluster.rep.insert(i, (tmpSet[i] + (merged_cluster.center - tmpSet[i]) * self.alpha))
            merged_cluster.rep = np.concatenate((merged_cluster.rep ,(tmpSet[i] + (merged_cluster.center - tmpSet[i]) * self.alpha)))

        return merged_cluster

    def union_func(self, cluster1, cluster2):
        union_cluster = Cluster(shape=self.shape)
        union_cluster.points=np.append(cluster1.points, cluster2.points, axis=0)
        return union_cluster

    def closest_cluster(self, cluster, merged_cluster, dist):

        distance = dist
        closest_rep = []

        for representative in cluster.rep:
            query = self.KDTree.query(representative, self.c+1, 0, 2, dist)

        for i in range(0, self.c+1):
            query_check = np.squeeze(np.asarray(cluster.rep))
            merged_check = np.squeeze(np.asarray(merged_cluster.rep))
            temp_rep = self.KDTree.data[query[1][0][i]]

            if (query[0][0][i] < distance and not (query_check == temp_rep).all() and not (merged_check == temp_rep).all()):
                distance = query[0][0][i]
                closest_rep = temp_rep

        # for clusterz in self.Heap:
        # for point in clusterz.rep:
        # if (point == closest_rep).all():
        # return (distance, clusterz)

        for clusterz in self.Heap:
            for point in clusterz.rep:
                tet = np.squeeze(np.asarray(point))
                if (tet == closest_rep).all():
                    return (distance, clusterz)

    def distance_func(self, p1, p2):
        return distance.euclidean(p1, p2)


def __load_file(path):
    """
    Load a data set from path. Data set must be arff format.
    
    :param path: path to the data set
    :return: a numpy-matrix. each column represents an attribute; each row a data item
    """
    data, meta = arff.loadarff(open(path, 'r'))
    if data.shape == (0,):
        return np.empty((0, len(meta._attributes))), 0
    else:
        data_matrix = np.zeros(shape=(data.shape[0], len(data[0])-1))

        for i in range(len(data)):
            arff_row = data[i]

            for j in range(len(arff_row)-1):
                data_matrix[i][j] = arff_row[j]

        return data_matrix, data.shape[0]


if __name__ == '__main__':
    file_name = str(sys.argv[1])
    number_of_clusters = int(sys.argv[2])
    alpha = float(sys.argv[3])
    c = int(sys.argv[4])

    data, length = __load_file(file_name)

    cure = Cure(data, number_of_clusters, alpha, c)
    list_of_labels = cure.cure_clustering()

    print(str(list_of_labels))