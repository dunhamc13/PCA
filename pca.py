#if using jupyter notebook uncomment next line
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import pandas as pd

#Save this in case needed later
import seaborn as sns; sns.set()
#from IPython import get_ipython
#get_ipython().run_line_magic(('matplotlib','inline')

#First create a random state object to get a random stream of numbers
#later we use this with rand() to 
rng = np.random.RandomState(1)

#Use randn() to generate a normal bell shaped distrbution, rand() creates
#a uniform distribution.
#np.dot() accepts two array like arguments and returns nDimensional array
nArray = np.dot(rng.rand(2,2), rng.randn(2,1000)).T

#Get the mean for reconstruction
#Xmean = np.mean(nArray, axis = 0)
print('the mean is ', mean, '\n')

#Check the dimension of the array
nDim = nArray.ndim
print('The array has ', nDim, ' Dimensions.\n')

#Check the number of objects in the array / remember each x,y pair is one object
#so you must divide by 2
nObj = nArray.size
nObj = nObj / 2
print('The array has ', nObj, ' data points.\n')


#Check the shlape of the array
nShape = nArray.shape
print('The array has a shape of ', nShape, ' (Rows, Columns).\n')

#Final step is to use matplotlib to create scatter plot
#making the scaling equal and then showing it.
#plt.scatter(nArray[:,0], nArray[:,1])
#plt.axis('equal');
#plt.show()


################################################################################
# Step 2 get the first pca
# ##############################################################################

#First pre-process data - scale it so nothing is more important
nArray_Scaled = StandardScaler().fit_transform(nArray)
nArray_Scaled[:2]

#Next get covariance
dimensions = nArray_Scaled.T
#dimensions = nArray.T
#dimensions_meaned = dimensions - np.mean(dimensions, axis = 0)
covMatrix = np.cov(dimensions)
covMatrix[:2]

#Then get eigen values and vectors
eigVal, eigVec = np.linalg.eig(covMatrix)
eigVal[:2]
eigVec[:2]

#Explained variance per component calculation
explVar = []
for i in range(len(eigVal)):
   explVar.append(eigVal[i] / np.sum(eigVal))

#first value should be 1
#The first column should be an array of the percentage of explained variances
#in pca 1 - hope it's over 90%
print(np.sum(explVar), '\n', explVar)

#get pca vals by using array dot product of the eigen vector and the scaled data
pca1 = nArray_Scaled.dot(eigVec.T[0])
#pca1 = dimensions.dot(eigVec.T[0])
#pca1 = dimensions.dot(eigVec)
#result = pd.DataFrame(pca1, columns=['PCA1'])
#result.head()

#plt.figure(figsize=(20,10))
#sns.scatterplot(result['PCA1'], [0] * len(result)) 

plt.scatter(pca1,[0]*len(pca1))
plt.axis('equal')
plt.show()



################################################################################
# Step 2 get the first pca
# ##############################################################################
