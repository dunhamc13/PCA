#if using jupyter notebook uncomment next line
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import pandas as pd
from numpy import array, newaxis

#Save this in case needed later
import seaborn as sns; sns.set()
#from IPython import get_ipython
#get_ipython().run_line_magic(('matplotlib','inline')

#Creates an npArray object has 2 dimensions and 1000 objects
def X():
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

   return nArray


################################################################################
# Step 2 get the first pca
# ##############################################################################
#Gets the PCA scores of an npArray, with number of components desired
def PCA(X , num_components):
     
    #Step-1
    X_meaned = X - np.mean(X , axis = 0)
     
    #Step-2
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    #Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Step-6
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
    #Show PCA 
    #plt.scatter(X_PCA,[0] * len(X_PCA))
    #plt.axis('equal')
    #plt.show()
    return X_reduced


################################################################################
# Step 3 PCA Reconstruction
# ##############################################################################
#Takes PCA scores and projects original npArray
def PCAReconstruction(X , num_components):
     
    #First get the mean of X
    X_meaned = X - np.mean(X , axis = 0)
     
    #Second get the covariance matrix
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    #THird get the eigenvectors
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Sort the eigenvectors
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Use the largest eigenvector subset
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Get PC
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()

    #print(eigen_vectors.transpose().shape)
    #print(X_reduced.shape)
    
    #Final step - reconstruct - X @ V^T + the mean = XR
    X_PCARM = X_reduced @ eigen_vectors.T + X_meaned
    plt.scatter(X_PCARM[:,0], X_PCARM[:,1])
    plt.axis('equal')
    plt.show()
    return X_PCARM

def main():
   
   #make an 2d array of 1000 objects
   global X
   X = X()
   
   #get the PC score of that array
   X_PCA = PCA(X,1)
   
   #reconstruct PCA
   X_Reconstructed = PCAReconstruction(X,1)
   return

main()
