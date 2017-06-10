import random
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.io import arff
from os import listdir
from os.path import isfile, join

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
    def __init__(self, data, number_of_clusters, alpha):

        # data is a Numpy-array with rows as points and k
        # is the number of clusters
        self.data = data
        self.k = number_of_clusters
        self.alpha = alpha
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
        self.Heap = sorted(data_as_clusters, key=lambda x:
            x.distance_closest, reverse=False)

    def cure_clustering(self):
        while (len(self.Heap) > self.k):

            # Select an arbitrary cluster from heap
            cluster_u = self.Heap[0]
            cluster_v = cluster_u.closest
            

            # remove to be merged elements from the heap and resort heap
            remove_element(self.Heap, cluster_v)
            remove_element(self.Heap, cluster_u)

            # merge the clusters to form a new cluster
            cluster_w = merge_cluster(cluster_u, cluster_v)

            # TODO: delete rep entries from kdtree and add w representative

            # select arbitrary element from the heap
            cluster_w.closest = self.Heap[0]
            cluster_w.distance_closest = distance_func(cluster_w.center,
                cluster_w.closest.center)

            for cluster in self.Heap:

                dist = distance_func(cluster_w.center, cluster.center)

                if (dist < cluster_w.distance_closest):

                    cluster_w.closest = cluster
                    cluster_w.distance_closest = dist

                if ((cluster.closest is cluster_u) or (cluster.closest is cluster_v)):

                    if(cluster.distance_closest < dist):
                        # get closest element to cluster with maximum distance
                        # dist TODO: check this
                        (cluster.closest, cluster.distance_closest)=KDTree.query(
                            cluster.points[0], 1, 0, 2, dist)

                    if (cluster.closest is None):
                        cluster.closest=cluster_w
                        cluster.distance_closest=dist

                    else:
                        cluster.closest=cluster_w
                        cluster.distance_closest=distance

                elif (cluster.distance_closest > dist):
                    cluster.closest=cluster_w
                    cluster.distance_closest=dist

            self.Heap.append(cluster_w)
            self.Heap.sort(key=lambda x: x.distance_closest, reverse=False)

    def merge_cluster(self, cluster1, cluster2):
        # TODO: add merge function from paper
        merged_cluster = union_func(self,cluster1,cluster2)
        merged_cluster.center=(len(cluster1.points)*cluster1.center+len(cluster2.points)*cluster2.center)/(len(cluster1.points)+len(cluster2.points))
        tmpSet
        for i in merged_cluster.rep:
            maxDist=0
            for point in merged_cluster:
               if  i == 1:
                minDist=distance_func(point,merged_cluster.center)
               else:     
                for pointQ in tmpSet:
                    tmpDist= distance_func(point,pointQ)
                    if minDist>= tmpDist:
                        minDist=tmpDist

                if minDist >= maxDist:
                    maxDist = minDist
                    maxPoint=pointQ

            tmpSet.point[len(tmpSet.point)]=maxPoint
           
        for point in tmpSet:
            #Smaller alpha shrinks the scattered points and favors elongated clusters
            #large alph-> scattered points get closer to mean,  cluster tend to be more compact
            merged_cluster.rep[len(merged_cluster.rep)]=point+(merged_cluster.center-point)*self.alpha
   
        return merged_cluster


    def remove_element(self, arr, element):
        index=arr.index(element)
        arr[index:index + 1]=[]

    
    def union_func(self,cluster1,cluster2):
        union_cluster
        unionpointer=len(cluster1.points)
        for i in xrange(0,len(cluster1.points)-1):
            union_cluster.point[i]=cluster1.point[i]
        for i in xrange(0,len(cluster2.points)-1):
            union_cluster.point[unionpointer+i]=cluster2.point[i]        
        return union_cluster

              

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
        return numpy.empty((0, len(meta._attributes))), 0
    else:
        return data[meta.names()[:-1]].reshape(data.shape+(-1,)), data.shape[0]


if __name__ == '__main__':
    text="the quick brown fox jumped over the the quick brown quick log log"

    file_name = str(sys.argv[1])
    number_of_clusters = int(sys.argv[2])
    alpha = float(sys.argv[3])

    data,length = __load_file(file_name)
    print data
    np.matrix(data).astype(np.float)
    print data
    cure = Cure(data, number_of_clusters, alpha)
    list_of_labels = cure.cure_clustering()

    print(str(list_of_labels))
