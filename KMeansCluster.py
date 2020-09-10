import pandas as pd
import numpy as np

#

class KMeansCluster():
    def __init__(self, NumIters, K):
        self.K = K
        self.Iters = NumIters
        self.Centroid = None
        self.EuclidianDistance = None
        self.Cluster = None
        self.Inertia = 0
        
    def fit(self, X_Data):
        X = X_Data.values #Convert to array
        
        M=len(X) #Number of Samples
        N=len(X[0]) #Number of Features
        
        self.Centroid = np.random.rand(self.K, N)
        
        
        for i in range(self.Iters):
            self.EuclidianDistance = np.array([]).reshape((M,0)) #Initialize Euclidian Distance Matrix
            
            for j in range(self.K):
                Distance = np.sum((X-self.Centroid[j])**2, axis=1) #Calculate Distance
                self.EuclidianDistance = np.c_[self.EuclidianDistance,Distance]

            self.Cluster = np.argmin(self.EuclidianDistance,axis=1)+1 #Cluster Is Closest Centroid



            Y = {} #Initialize current cluster index

            for k in range(self.K):
                Y[k+1] = np.array([]).reshape(N,0) #Populate dict with K arrays

            for x in range(M): 
                Y[self.Cluster[x]]=np.c_[Y[self.Cluster[x]], X[x]] #Populate arrays with co-ordinates of cluster members
                    

            for k in range(self.K):
                Y[k+1]=Y[k+1].T #Columns to rows

            for k in range(self.K):
                self.Centroid[k]=np.mean(Y[k+1],axis=0) #Cluster Centroid is Mean of Cluster Member Co-ordinates

            Nans = np.isnan(self.Centroid) #Reset Nan centroids to random centroid (First round)
            self.Centroid[Nans] = np.random.rand()
            
        return self
    
    
    def Predict(self, X_Data):
        X = X_Data.values #Convert to array
        
        for j in range(self.K):
            Distance = np.sum((X-self.Centroid[j])**2, axis=1) #Calculate Distance
            self.EuclidianDistance = np.c_[self.EuclidianDistance,Distance]

        self.Cluster = np.argmin(self.EuclidianDistance, axis=1)+1 #Cluster Is Closest Centroid

        self.Inertia = np.sum(np.absolute(np.min(self.EuclidianDistance, axis=1)))
        
        return self.Cluster