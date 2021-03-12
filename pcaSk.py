import numpy as np
import sklearn.datasets, sklearn.decomposition
import scipy
from sklearn.feature_extraction import image
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mping
import pandas as pd

#First create a random state object to get a random stream of numbers
#later we use this with rand() to 
rng = np.random.RandomState(1)

#Use randn() to generate a normal bell shaped distrbution, rand() creates
#a uniform distribution.
#np.dot() accepts two array like arguments and returns nDimensional array
X = np.dot(rng.rand(2,2), rng.randn(2,1000)).T

images = loadmat('USPS.mat')
imgplot = plt.imshow(images[:,:,0])
plt.show();
#print(images.shape)


#plt.scatter(X[:,0], X[:,1])
#plt.axis('equal');
#plt.show()
#X = sklearn.datasets.load_iris().data
mu = np.mean(X, axis=0)

pca = sklearn.decomposition.PCA()
pca.fit(X)
X_pca = pca.transform(X)
#plt.scatter(X_pca[:,0], [0]*X_pca[:,1])
#plt.axis('equal');
#plt.show()

nComp = 1
Xhat = np.dot(pca.transform(X)[:,:nComp], pca.components_[:nComp,:])
Xhat += mu
plt.scatter(Xhat[:,0], Xhat[:,1])
plt.axis('equal');
#plt.show()

#print(Xhat[0,])
