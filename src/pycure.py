import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from scipy.spatial import distance


class Cluster:
    def __init__(self, point=None):

        if (point is not None):
            self.points = [point]
            self.center = point
            self.rep = [point]

        else:
            self.points = []
            self.center = None
            self.rep = []

        self.closest = None
        self.distance_closest = float('inf')


class Cure:
    def __init__(self, data, number_of_clusters):

        # data is a Numpy-array with rows as points and k is the number of
        # clusters
        self.data = data
        self.k = number_of_clusters

        # Stores representatives for each cluster
        self.KDTree = KDTree(data)

        # Initializes each point as a Cluster object
        data_as_clusters = [Cluster(point) for point in data]

        # Initializes each Clusters closest Cluster and distance using the
        # KDTree
        for cluster in data_as_clusters:
            query = KDTree.query(cluster.points[0])
            cluster.distance_closest = query[0]
            cluster.closest = data_as_clusters[query[1]]

        # Stores an entry for each cluster sorted by their distances to their
        # closest cluster
        self.Heap = sorted(data_as_clusters,
                           key=lambda x: x.distance_closest,
                           reverse=False
                           )

    def cure_clustering(self):
        while (len(self.Heap) > self.k):

            # Select an first cluster from heap
            cluster_u = self.Heap[0]
            cluster_v = cluster_u.closest

            # remove to be merged elements from the heap and resort heap
            self.Heap.remove(cluster_v)
            self.Heap.remove(cluster_u)

            # merge the clusters to form a new cluster
            cluster_w = self.merge_cluster(cluster_u, cluster_v)

            # TODO: delete rep entries from kdtree and add w representative

            # select arbitrary element from the heap
            cluster_w.closest = random.choice(self.Heap)
            # cluster_w.closest = self.Heap[0]
            cluster_w.distance_closest = self.distance_func(
                                                       cluster_w.center,
                                                       cluster_w.closest.center
                                                       )

            for cluster in self.Heap:

                dist = self.distance_func(cluster_w.center, cluster.center)

                if (dist < cluster_w.distance_closest):

                    cluster_w.closest = cluster
                    cluster_w.distance_closest = dist

                if ((cluster.closest is cluster_u) or
                        (cluster.closest is cluster_v)):

                    if(cluster.distance_closest < dist):
                        # get closest element to cluster with maximum distance
                        # dist TODO: check this
                        (cluster.closest,
                         cluster.distance_closest) = KDTree.query(
                            cluster.points[0], 1, 0, 2, dist)

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

    def merge_cluster(self, cluster1, cluster2):
        merged_cluster = None
        # TODO: add merge function from paper
        return merged_cluster

    def remove_element(self, arr, element):
        index = arr.index(element)
        arr[index:index + 1] = []

    def distance_func(self, p1, p2):
        return distance.euclidean(p1, p2)

if __name__ == '__main__':
    text = "the quick brown fox jumped over the the quick brown quick log log"
    print(text)
