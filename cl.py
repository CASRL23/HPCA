# K-Means Clustering

# Importing the libraries

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import silhouette_score

import statsmodels.api as sm
from pca import PCA_Obj
from hierarchical import hierarchical
#Importing the mall dataset with pandas
# data = pd.read_csv('categorized_arria2.csv')
arria = PCA_Obj('updated_arria.csv')
# X = data.iloc[:,1:].values
X = hierarchical(arria)
# data = arria.x_old

for cluster_n,data in enumerate(X.clustered):
      from sklearn.cluster import KMeans
      max_clusters =11
      max_features =23
      wcss = [] #[[0 for x in range(max_clusters-1)] for y in range(max_features-1)]
      Sil = [] #[0 for x in range(max_clusters)] for y in range(max_features-1)]
      for j in range (2,max_features):
            col1 = []
            col2 = []
            new_x = data[:,:j] #[1,2]].values
            for i in range (1,max_clusters):
                  kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter =300, n_init = 10, random_state = 42)
                  label=kmeans.fit_predict(new_x) #X)
                  if i>1: print(f'SilouetteScore (n={i}): {silhouette_score(new_x,label)}')
                  col1.append(kmeans.inertia_)
                  if i>1: col2.append(silhouette_score(new_x,label))
            wcss.append(col1)
            Sil.append(col2)

      plt.plot(range(1,max_clusters),col1)


      plt.plot(range(1,max_clusters),col1)
      plt.title('')
      plt.xlabel('Clusters')
      plt.ylabel('Weighted sum of squared distances')
      plt.legend('cycles','clock period')
      # if cluster_n == 0:
      #       cat = "Control"
      # elif cluster_n == 1:
      #       cat = "Memory"
      # elif cluster_n == 2:
      #       cat = "Registers"
      # elif cluster_n == 3:
      #       cat = "ILP"
      # elif cluster_n == 4:
      #       cat = "IO"
      plt.savefig(f'unsupervised_cluster{cluster_n}_elbow.png')
      plt.clf()
      plt.plot(range(2,max_clusters),col2)
      plt.title('')
      plt.xlabel('Clusters')
      plt.ylabel('Silhouette Score')
      plt.savefig(f'unsupervised_cluster{cluster_n}_sil.png')
      plt.clf()
