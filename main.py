import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from helperFunctions import *
pd.set_option("display.max_rows", None, "display.max_columns", None)

##########################################################
############## Data Import ###############################
##########################################################

filepath = 'SandwichesAndFeatures_Minimal.csv' #location of data set
df = pd.read_csv(filepath,index_col='Food') #read data set into dataframe
df_orig = pd.read_csv(filepath,index_col='Food') #copy data set

for (columnName, columnData) in df.iteritems(): #Normalize/Scale column data
    df[columnName] = columnData.values/np.max(columnData.values) #divide each value in a column by the column max.

##########################################################
############## Hierarchical Clustering ###################
##########################################################

hierarchy = shc.linkage(df, method='ward') #create hierarchical cluster using sklearn package
fig1 = plt.figure(figsize=(10, 6))
dend = shc.dendrogram(hierarchy, labels = df.index.values)# draw clustering as dendrogram

plt.title("Hierarchical Sandwich Clustering")
plt.xticks(rotation = 75)
plt.subplots_adjust(bottom=0.25)
fig1.savefig('Dendrogram.png')

##########################################################
############## Principal Component Analysis ##############
##########################################################

pca = PCA(n_components=4) #create PCA object (again from sklearn)
data_PCA = pca.fit_transform(df) #transform dataframe to principal component space
explained_variance = pca.explained_variance_ratio_ #get variance explained by PCA components
pc1, pc2, pc3, pc4 =  pca.components_ #get principal component vectors
df_pca = pd.DataFrame([pc1,pc2,pc3, pc4], columns=df.columns) #load principal component coordinates into new data frame

#plot PCA components
pcComponents1, (axa,axb) = plt.subplots(1,2, figsize = (12,5))
pcComponents2, (axc,axd) = plt.subplots(1,2, figsize = (12,5))
plotPCbar(pc1,df.columns,axa) #plot sorted bar chart of components defining PC1
axa.set_title('PC1: ' + str(explained_variance[0]))
plotPCbar(pc2,df.columns,axb)#plot sorted bar chart of components defining PC2
axb.set_title('PC2: ' + str(explained_variance[1]))
plotPCbar(pc3,df.columns,axc)#plot sorted bar chart of components defining PC3
axc.set_title('PC3: ' + str(explained_variance[2]))
plotPCbar(pc4,df.columns,axd)#plot sorted bar chart of components defining PC4
axd.set_title('PC4: ' + str(explained_variance[3]))
pcComponents1.autofmt_xdate(rotation=75)
pcComponents1.subplots_adjust(bottom=0.3)
pcComponents2.autofmt_xdate(rotation=75)
pcComponents2.subplots_adjust(bottom=0.3)

##########################################################
########## Sandwiches Plotted in PC Space ################
##########################################################

#define shorthand for principal component coordinates
x = data_PCA[:,0] #coordinates of each food in along PC1 axis
y = data_PCA[:,1] #coordinates of each food in along PC2 axis
z = data_PCA[:,2] #coordinates of each food in along PC3 axis
w = data_PCA[:,3] #coordinates of each food in along PC4 axis

#Plot
sandwichesPC1, (ax1, ax2) = plt.subplots(1,2,figsize = (12,5))

scatterLabel(ax1, x, y, df.index.values)
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')

scatterLabel(ax2, z, w, df.index.values)
ax2.set_xlabel('PC 3')
ax2.set_ylabel('PC 4')

##########################################################
########### k-means clustering ###########################
##########################################################

#since k-means clustering requires a pre-defined number of clusters, I will generate figures for n=2 to n=5

clusterLabels = [] #array to store cluster labeling
for numClusters in range(2,6): #loop over numClusters from 2 to 5

    kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(data_PCA) #perform K-means clustering on data_in PC space
    labels = kmeans.labels_ #get clustered labels
    clusterLabels.append(labels) #add to array

    #plot sandwiches in PC space, but now with colors corresponding to their k-means cluster
    fig, (axi, axj) = plt.subplots(1,2, figsize = (15,7))
    scatterLabel(axi, x, y, df.index.values, colors = label2color(labels))
    axi.set_title('Num Clusters:' + str(numClusters))
    axi.set_xlabel('PC1')
    axi.set_ylabel('PC2')
    scatterLabel(axj, z, w, df.index.values, colors=label2color(labels))
    axj.set_title('Num Clusters:' + str(numClusters))
    axj.set_xlabel('PC3')
    axj.set_ylabel('PC4')

#Show all plots
plt.show()


