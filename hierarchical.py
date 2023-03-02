from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import empirical_covariance
import numpy
from sklearn.cluster import KMeans
from pca import sheet, PCA_Obj
import copy
'''
        self.x_old - scaled transformed input data (n_samples, n_features)
        self.feature_names - input feature names (n_features)
        self.n_samples - number of input samples
        self.subclusters - array of number of subclusters to make for each cluster
        self.kept_variance- percentage of variance to keep in each subcluster

        self.subclustered_feature_names - data[cluster[subcluster[feature_name]]]
        self.subclustered_vectors - data[cluster[subcluster[pc[feature]]]] eigenvectors.
        self.subclustered_pc_variance - data[cluster[subcluster[pc]]] explained variance ratio
        self.kept_subclustered_pc_labels - data[cluster[subcluster[pc]]] labels of kept PCs
        self.kept_subclustered_vectors - data[cluster[subcluster]] kept eigenvectors index [pc,feature]
        self.kept_subclustered_pc_variance - data[cluster[subcluster]] kept explained variance ratio index [pc]

        self.clustered_feature_names - data[cluster[feature_name]]
        self.clustered - data[cluster[sample[feature]]]
        self.kept_clustered_vectors - data[cluster[pc[sample]]] kept eigenvectors
        self.clustered_pc_variance - data[cluster[pc]] kept explained variance ratio

        self.eigenvectors - data[pc[sample]] eigenvectors of output
        self.pc_variance - data[pc] explained variance ratio of output
        self.final_data - final output of hierarchical PCA (n_samples, n_new_features)
        self.normalized_final_data - normalized self.final_data (n_samples, n_new_features)
'''
class hierarchical():
    def __init__(self,pcaObj,subclusters=[],kept_variance=.75):
        self.x_old = pcaObj.x_old
        self.feature_names = pcaObj.feature_names
        self.n_samples = pcaObj.n_samples
        self.subclusters = subclusters
        self.kept_variance = kept_variance
        self.final_data = self.run()
    # given data to cluster, name of features to cluster, and number of clusters, return the clustered data & clustered feature names
    def cluster(self,to_cluster,n_clusters,featureNames,labels=None):
        hierarchical_cluster = KMeans(verbose=0,n_clusters=n_clusters,random_state=1)
        if labels is None:
            labels = hierarchical_cluster.fit_predict(numpy.transpose(to_cluster))
        sorted_labels = numpy.argsort(labels)
        clustered = numpy.split(numpy.transpose(to_cluster[:,sorted_labels]), numpy.unique(labels[sorted_labels], return_index=True)[1][1:])
        clustered = [numpy.transpose(i) for i in clustered]
        # sort the feature names into clusters same as the feature data itself
        clustered_feature_names = numpy.split(featureNames[sorted_labels], numpy.unique(labels[sorted_labels], return_index=True)[1][1:])
        return clustered,clustered_feature_names
    def run(self):
        #cluster the data

        #cluster labels
        # combined categorical
        # labels = numpy.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4])

        # yellow categorical IO
        # labels = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

        #blue categorical IO
        # labels = numpy.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4])

        #green categorical IO
        # labels = numpy.array([0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        #cluster labels

        if 'labels' not in locals():
            clustered,clustered_feature_names = self.cluster(self.x_old,7,self.feature_names)
        else:
            clustered,clustered_feature_names = self.cluster(self.x_old,7,self.feature_names,labels)
        self.clustered = clustered
        self.clustered_feature_names = clustered_feature_names
        new_clusters = self.subcluster(clustered)
        # var_pc_list = []
        kept_feature_variance = []
        for n_cluster,cluster in enumerate(self.kept_subclustered_vectors):
            cluster_feature_variance = []
            for n_sub,subcluster in enumerate(cluster):
                cluster_pc_variance = self.subclustered_pc_variance[n_cluster]
                sub_pc_variance = cluster_pc_variance[n_sub]
                temp_sub_feature_variance = numpy.empty((0,len(subcluster[0])))
                for i in range(len(subcluster)):
                    kept_feature_var = self.feature_variance(subcluster,sub_pc_variance,target_PC=i)

                    #makes an array with PCs as rows and features as columns for feature var in each PC
                    temp_sub_feature_variance = numpy.vstack([temp_sub_feature_variance,kept_feature_var])
                cluster_feature_variance.append(temp_sub_feature_variance)
            kept_feature_variance.append(cluster_feature_variance)
        if new_clusters == 0:
            new_clusters = clustered
        final_data = self.join_clusters(new_clusters)
        return final_data

    def subcluster(self,clustered):
        if len(self.subclusters) != len(clustered):
            print("Error: number of subclusters does not match number of clusters")
            return 0
        if self.subclusters != []:
            cluster_arr_list = []
            cluster_name_list = []
            # now that we have the data in clusters, make subclusters of the data into a new list, with corresponding feature names
            for l,i in enumerate(clustered):
                clusters = self.subclusters[l]

                subcluster,subcluster_feature_names = self.cluster(i,clusters,self.clustered_feature_names[l])
                cluster_arr_list.append(subcluster)
                cluster_name_list.append(subcluster_feature_names)
            subclustered_vectors = []
            kept_subclustered_vectors = []
            subcluster_pc_variance = []
            new_clusters = []
            kept_subclustered_pc_labels = []
            kept_subclustered_pc_variance = []
            # run through each cluster.
            for i in cluster_arr_list:
                subcluster_eigenvectors = []
                kept_subcluster_eigenvectors = []
                kept_subcluster_pc_variance = []

                sub_pc_variance = []
                kept_subcluster_pc_labels = numpy.empty(0)
                # make an empty array to make one array for the whole cluster
                new_subclusters = numpy.empty((self.n_samples,0))
                # run through each subcluster & apply PCA
                for c,l in enumerate(i):
                    # make a temp pca object and fit it to our subcluster
                    temp_pca= PCA()
                    new_sub_cluster = temp_pca.fit_transform(l)

                    counter = 0
                    varsum = 0
                    #goes through the subcluster, and only keeps enough components to explain at least 75% of the dataset
                    for x in temp_pca.explained_variance_ratio_:
                        varsum += x
                        counter += 1
                        if varsum >= self.kept_variance:
                            break
                    new_sub_cluster = new_sub_cluster[:,0:counter]
                    # stack the new sub cluster onto 'new subclusters' as one dataset
                    new_subclusters = numpy.hstack([new_subclusters,new_sub_cluster])
                    # keep track of eigenvectors in each subcluster.
                    subcluster_eigenvectors.append(temp_pca.components_)
                    kept_subcluster_eigenvectors.append(temp_pca.components_[:,0:counter])
                    kept_subcluster_pc_variance.append(temp_pca.explained_variance_ratio_[0:counter])
                    # keep track of explained variance of each PC
                    sub_pc_variance.append(temp_pca.explained_variance_ratio_)

                    temp_pc_labels = [f"Subcluster {c} PC{p}" for p in range(len(new_sub_cluster[0]))]
                    kept_subcluster_pc_labels = numpy.hstack([kept_subcluster_pc_labels,temp_pc_labels])
                subclustered_vectors.append(subcluster_eigenvectors)
                kept_subclustered_vectors.append(kept_subcluster_eigenvectors)
                subcluster_pc_variance.append(sub_pc_variance)
                kept_subclustered_pc_variance.append(kept_subcluster_pc_variance)
                kept_subclustered_pc_labels.append(kept_subcluster_pc_labels)

                #a new array of clusters, from the reduced subclusters (column L)
                new_clusters.append(new_subclusters)
            self.subclustered_vectors = subclustered_vectors
            self.kept_subclustered_vectors = kept_subclustered_vectors
            self.subclustered_pc_variance = subcluster_pc_variance
            self.kept_subclustered_pc_labels = kept_subclustered_pc_labels
            self.subclustered_feature_names = cluster_name_list

        return new_clusters
    def retained_variance(self):
        #get the retained variance of each subcluster
        subcluster_retained_variance = []
        for i in self.kept_subclustered_vectors:
            feature_variance = 
            subcluster_retained_variance.append([sum(x) for x in i])
        self.subclustered_retained_variance = subcluster_retained_variance
        return subcluster_retained_variance

    def join_clusters(self,new_clusters):
        # to check what the data from each cluster was, for output
        combined_clusters = []
        combined_data = numpy.empty((self.n_samples,0))
        cluster_vectors = []
        combined_pc_var = []
        #run PCA on each new cluster
        for j in new_clusters:
            temp_pca = PCA()
            new_cluster = temp_pca.fit_transform(j)
            new_cluster_vectors = temp_pca.components_
            counter = 0
            varsum = 0
            for x in temp_pca.explained_variance_ratio_:
                varsum += x
                counter += 1
                if varsum >= self.kept_variance:
                    break
            new_cluster = new_cluster[:,0:counter]
            combined_clusters.append(new_cluster)
            #get the final data. Column Q
            combined_data = numpy.hstack([combined_data,new_cluster])
            cluster_vectors.append(new_cluster_vectors[0:counter])
            combined_pc_var.append(temp_pca.explained_variance_ratio_[0:counter])
        final_pca = PCA()
        final_data = final_pca.fit_transform(combined_data)
        final_data_scaler = StandardScaler()
        #normalize the final data
        #give the scaler object information from final data
        final_data_scaler.fit(final_data)
        #normalize (z-score) the data's columns
        normalized_final_data=final_data_scaler.transform(final_data)
        # return cluster_vectors, combined_pc_var, kept_subclustered_pc_labels

        self.normalized_final_data = normalized_final_data
        self.eigenvectors = final_pca.components_
        self.pc_variance = final_pca.explained_variance_ratio_
        self.kept_clustered_vectors = cluster_vectors
        self.clustered_pc_variance = combined_pc_var

        return final_data

    def feature_variance(self,vectors,PC_variance,target_PC=None):
        scaled_evectors = abs(copy.deepcopy(vectors))
        #scales eigenvectors relative to their PC explained variance, useful for finding explained var by feature
        for x in range(len(vectors)):
            total = numpy.sum(scaled_evectors[x])
            scaled_evectors[x] = numpy.multiply(scaled_evectors[x],PC_variance[x])
            scaled_evectors[x] = numpy.divide(scaled_evectors[x],total)
        var_total = numpy.empty(len(vectors[0]))
        #gets explained variance of all features for all PCs & input PC
        if target_PC is None:
            for y in range(len(vectors[0])):
                var_total[y] = scaled_evectors[:,y].sum()
        else:
            for y in range(len(vectors[0])):
                var_total[y] = scaled_evectors[target_PC,y]/PC_variance[target_PC]
        return var_total


