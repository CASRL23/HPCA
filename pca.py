'''
March 19, 2021
Author: Grant Deljevic, grant.n.deljevic@wmich.edu
Property of Western Michigan University
'''
import enum
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import empirical_covariance
import numpy
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class PCA_Obj():
    '''
    self.x_old: Normalized data. Rows are samples columns are features. Shape (n_samples,n_features)
    self.x_new: New data after PCA has been applied. Rows are samples columns are Principal Components. Shape (n_samples,n_pcs)
    self.components: Eigenvectors of data. Rows are PCs, columns are features. Multiplied by x_old to get x_new. Shape (n_pcs,n_features)
    self.PC_total:
      If target_pc is given, returns an array of shape (1,n_features) of each feature's explained variance in the target PC.
      If target_pc is not given, returns an array of shape (1,n_features) of each feature's total explained variance
    self.plot: shows a plot of the features in 3 k-means clusters.
    self.sample_names: The data sample (row) names
    self.feature_names: The data feature (column) names
    self.pca: returns the sklearn PCA object as it is at the end of PCA.
    self.emp_cov: Returns emperical (estimated) covariance of the scaled data
    self.PC_variance: An array of shape (1,n_pcs) which shows the explained variance of each PC
    self.n_pcs: Number of principal components after PCA has been applied.
    self.n_features: Number of features (columns) in the data
    self.n_samples: Number of samples (rows) in the data
    self.PC_labels: the PC labels PC1-PC{n_pcs}
    self.transposed: returns a new instance of the PCA_Obj with the rows used as features and the columns used as samples.
      Has all of the above attributes.
    self.plot_new_benchmarks(): plots each sample's new values projected in a 2d space where x = PC1 and y = PC2 (first 2 cols of self.x_new)
    self.plot_feature_vectors(): plots each feature's eigenvectors projected in a 2d space where x = PC1 and y = PC2 (first 2 rows of self.components)
      Distance along axis is total explained variance, points in the same direction are positively correlated, points in the opposite direction are negatively correlated.
    '''
    def __init__(self,filename,transposed=False,target_PC=None):

        #filename: the file name (including extension) that you wish to apply PCA to
        #transposed: defaults to false. Should not be specified in normal use, run as normal and use self.transposed instead.
        #target_PC: defaults to None. Determines which PC total is returned for self.PC_total. If None, take sum of all PCs.
        self.x_old = self.process(filename,transposed)
        self.x_new = self.apply_pca()
        self.PC_total = self.feature_variance(target_PC)
        # self.plot_feature_vectors = self.biplot(self.x_new[:,0:2])
        # self.plot_new_data = self.biplot(numpy.transpose(self.components[0:2, :]))
        if transposed == False:
            self.transposed = PCA_Obj(filename,transposed=True)
    #process the given file into a useable, normalized dataset
    def process(self,filename,transposed):
        features = pd.read_csv(filename)
        if transposed:
            #returns the samples. First value in each row.
            self.sample_names = features.columns[1:].values.flatten()
            #returns the features. First value in each column
            self.feature_names = features.iloc[:,:1].values.flatten()
        else:
            #if transposed, return row labels as feature names & col names as samples
            self.feature_names = features.columns[1:].values.flatten()
            self.sample_names = features.iloc[:,:1].values

        #data without sample labels
        features = features.iloc[:,1:]
        #select only data without feature labels or row indexes
        X = features.values

        scaler = StandardScaler()
        #give the scaler object information from X
        scaler.fit(X)

        #normalize (z-score) the data's columns
        X=scaler.transform(X)

        #if input, turn X on its side to treat the rows as features and the columns as samples
        if transposed:
            X = X.transpose()

        return X

    def apply_pca(self):
        self.pca = PCA()

        #dot product of the eigenvectors for every feature in each PC, with each row of x_old
        #The original data transformed by PCA. Rows are samples, columns are PCs.
        x_new = self.pca.fit_transform(self.x_old)

        #emperical(estimated) covariance of scaled data - assume centered is from dividing by mean
        self.emp_cov = empirical_covariance(self.x_old,assume_centered=True)

        #the percent of explained variance of each principal component in the new data. has n_PCs values, sums to 1(100%)
        self.PC_variance = self.pca.explained_variance_ratio_
        #eigenvectors of the new data -- the direction they're being projected & by how much
        self.components = self.pca.components_
        #number of PCs in obj — min(n_features,n_samples)
        self.n_pcs = self.pca.n_components_
        #number of features in obj
        self.n_features = self.pca.n_features_
        #number of samples in obj
        self.n_samples = self.pca.n_samples_

        #list of PC labels for data
        self.PC_labels = [f'PC{i+1}' for i in range(self.n_pcs)]
        return x_new

    def feature_variance(self,target_PC=None):
        scaled_evectors = abs(copy.deepcopy(self.components))
        #scales eigenvectors relative to their PC explained variance, useful for finding explained var by feature
        for x in range(self.n_pcs):
            total = numpy.sum(scaled_evectors[x])
            scaled_evectors[x] = numpy.multiply(scaled_evectors[x],self.PC_variance[x])
            scaled_evectors[x] = numpy.divide(scaled_evectors[x],total)
        var_total = numpy.empty(self.n_features)
        #gets explained variance of all features for all PCs & input PC
        if target_PC is None:
            for y in range(self.n_features):
                var_total[y] = scaled_evectors[:,y].sum()
        else:
            for y in range(self.n_features):
                var_total[y] = scaled_evectors[target_PC,y]/self.PC_variance[target_PC]
        return var_total

    def biplot(self, score):
        #cluster details
        hierarchical_cluster = KMeans(verbose=0,n_clusters=7)

        '''
        Original Author: Serafeim Loukas, serafeim.loukas@epfl.ch
        Modified by Grant Deljevic
        Inputs:
            score: the projected data
            coeff: the eigenvectors (PCs)
        y: the class labels
         '''
        xs = score[:,0] # projection on PC1
        ys = score[:,1] # projection on PC2
        labels = hierarchical_cluster.fit_predict(list(zip(xs,ys)))
        plt.figure(figsize=(10,8), dpi=100)
        if 'labels' in locals():
            plt.scatter(xs,ys, c = labels) # color based on group
        else:
            plt.scatter(xs,ys)
        if len(xs) == self.n_samples:
            for i, label in enumerate(self.sample_names):
                plt.annotate(label, (xs[i], ys[i]),fontsize=6)
        elif len(xs) == self.n_features:
            for i, label in enumerate(self.feature_names):
                plt.annotate(label, (xs[i], ys[i]),fontsize=6)

        plt.xlabel("PC{}".format(1), size=14)
        plt.ylabel("PC{}".format(2), size=14)
        limx= int(xs.max()) + numpy.std(xs)*2
        limy= int(ys.max()) + numpy.std(ys)*2
        plt.xlim([-limx,limx])
        plt.ylim([-limy,limy])
        plt.grid()
        plt.tick_params(axis='both', which='both', labelsize=9)
        return plt.show()
    def plot_new_benchmarks(self):
        self.biplot(self.x_new[:,0:2])
    def plot_feature_vectors(self):
        self.biplot(numpy.transpose(self.components[0:2, :]))
def feature_variance(vectors,PC_variance):
    scaled_evectors = abs(copy.deepcopy(vectors))
    #scales eigenvectors relative to their PC explained variance, useful for finding explained var by feature
    for x in range(len(vectors)):
        total = numpy.sum(scaled_evectors[x])
        scaled_evectors[x] = numpy.multiply(scaled_evectors[x],PC_variance[x])
        scaled_evectors[x] = numpy.divide(scaled_evectors[x],total)
    var_totals = numpy.empty(len(vectors[0]))
    #gets explained variance of all features for all PCs & input PC
    for y in range(len(vectors[0])):
        var_totals[y] = scaled_evectors[:,y].sum()
    return var_totals
