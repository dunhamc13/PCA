import numpy as np 
import math
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import scipy.io as scio

'''
   load_data(matData_loc)
   matData_loc is the location of the mat file
   creates a dictionary objects and returns it
'''
def load_data(matData_loc):
    data = scio.loadmat(matData_loc)
    return data['A']

'''
   save_image(X_reconstructed,pc, i)
   takes the reconstructed matrix and saves it as png
   X_reconstructed : matrix that has been reconstructed
   pc is the number of principle compoents for the reconstruction
   i is a variable to choose which image from matrix to save
'''
def save_images(X_reconstructed, pc, i):
    fig = plt.figure()
    Xr = X_reconstructed
    plt.imshow(Xr[i,:].reshape((16,16)),cmap='gray')
    fig.savefig('image%d_features%d'%(i,pc))
    
'''
   reconstruction error(X_raw, X_reconstructed)
   X_raw : original data matrix 3000, 256
   X_reduced : the reduced matrix witha sahpe of (3000, 256)
   Return is the error using mean squared.
'''
def reconstruction_error(X_raw, X_reconstructed):
        #r2 = r2_score(X_raw, X_reconstructed)
        #rmse = mean_squared_error(X_raw, X_reconstructed)
        rmse = ((X_raw - X_reconstructed)**2).mean()
        #rmse = sqrt(mean_squared_error(X_raw, X_reconstructed))
        #nrmse = rmse/sqrt(np.mean(X_raw**2))
        return rmse

'''
  pca_reconstruction(num_pc, X_reduced)
  Args:
  num_pc: the number of principal components in matrix shaped (3000, p).
  X_reduced: The reduced matrix after PCA with shape (3000, p).
  Returns:
  X_reconstructed: The reconstructed matrix with shape (3000, 256)
'''
def pca_reconstruction(num_pc, X_reduced):
        #return np.dot(num_pc, X_reduced)
        X_new = pca.inverse_transform(X_Reduced)
        return X_new

'''
   pca(X_raw,p)
   pca uses the svd method to calculate PCA
   X_raw is the matrix from the mat file : shape is (3000, 256) or 3000 objects
      with 256 features.
   pc is the number of principle compoents to reduce down too
   Returns num_pc from matrix (3000,pc)
      and X_reduced : the matrix after PCA with a shape (pc, 3000)
'''
def pca(X_raw, pc):
        
        #Step 1 get the sample mean
        #X_bar = np.mean(X_raw, axis=0).reshape(1, X_raw.shape[1])

        #Step 2 center the data
        #centered = X_raw - np.dot(np.ones((X_raw.shape[0], 1)), X_bar)

        #Step 3 use svd to get u: array of ndim, s vector, and vh array ndim
        #u, s, vh = np.linalg.svd(centered)
        #return(u[:,:pc], np.dot(np.transpose(u[:,:pc]), X_raw))
        pca = PCA(pc)
        pca.fit(X_raw)
        #print(pca.explained_variance_)
        X_pca = pca.transform(X_raw)
        X_new = pca.inverse_transform(X_pca)
        return X_new
'''
   run_pca()
   accepts no arguments
   loads the USPS.mat file and converts it to a matrix.
   Next, runs principle compoents 10,50,100, and 200 to compare 
   image reconstruction.  All files are saved in directory.
   Finally, reconstruction error is computed.
'''
def run_pca():
    #Variables : ps is an array of the principle components used for reconstruction
    num_pc_array = [10, 50, 100, 150, 200]
    
    #Step 1 load the USPS.mat file
    data_loc = "./USPS.mat"

    #Step 2 convert the mat file to a matrix
    X_raw = load_data(data_loc)

    #Save the last 2 images from the orignal data set for comparison (off by 1)
    save_images(X_raw, 256, 2998)
    save_images(X_raw, 256, 2999)
    
    #Step 3 loop through ps array to run PCA / Reconstruction / error
    for pc in num_pc_array:

        #Step 1 get pca
        #num_Pc, X_reduced = pca(X_raw, pc)
        X_reconstructed = pca(X_raw, pc)

        #Step 2 do PCA reconstruction
        #X_reconstructed = pca_reconstruction(num_Pc, X_reduced)

        #Step 3 get reconstruction error
        error = reconstruction_error(X_raw, X_reconstructed)

        #Step 4 print the error and save images for first 16
        print("Error rate is %.02f%% for %d features" %((error*100), pc))
        save_images(X_reconstructed, pc, 2998)
        save_images(X_reconstructed, pc, 2999)
    
if __name__ == '__main__':
    run_pca()
