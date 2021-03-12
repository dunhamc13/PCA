import numpy as np
import matplotlib.pyplot as plt
import scipy.io
mat = scipy.ioloadmat('USPS.mat')

#Creates an npArray object has 2 dimensions and 1000 objects
def X():
   #First create a random state object to get a random stream of numbers
   #later we use this with rand() to 
   rng = np.random.RandomState(1)

   #Use randn() to generate a normal bell shaped distrbution, rand() creates
   #a uniform distribution.
   #np.dot() accepts two array like arguments and returns nDimensional array
   x_raw = np.dot(rng.rand(2,2), rng.randn(2,1000)).T

   #Get the mean for reconstruction
   #Xmean = np.mean(nArray, axis = 0)
   #print('the mean is ', mean, '\n')

   #Check the dimension of the array
   nDim = x_raw.ndim
   print('The array has ', nDim, ' Dimensions.\n')

   #Check the number of objects in the array / remember each x,y pair is one object
   #so you must divide by 2
   nObj = x_raw.size
   nObj = nObj / 2
   print('The array has ', nObj, ' data points.\n')


   #Check the shlape of the array
   nShape = x_raw.shape
   print('The array has a shape of ', nShape, ' (Rows, Columns).\n')

   #Final step is to use matplotlib to create scatter plot
   #making the scaling equal and then showing it.
   plt.scatter(x_raw[:,0], x_raw[:,1])
   plt.axis('equal');
   plt.show()

   return X



#Data Centralization
def Z_centered(dataMat):
    rows,cols=dataMat.shape
    meanVal = np.mean(dataMat, axis=0)  # Find the mean by column, that is to say, the mean of each feature.
    meanVal = np.tile(meanVal,(rows,1))
    newdata = dataMat-meanVal
    return newdata, meanVal

#covariance matrix
def Cov(dataMat):
    meanVal = np.mean(data,0) #Compress rows, return 1*cols matrix, average columns
    meanVal = np.tile(meanVal, (rows,1)) #Returns the mean matrix of rows rows
    Z = dataMat - meanVal
    Zcov = (1/(rows-1))*Z.T * Z
    return Zcov
    
#Minimize the loss caused by dimensionality reduction and determine k
def Percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)  # Ascending order
    sortArray = sortArray[-1::-1]  # Reversal, i.e. descending order
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num
    
#Get the largest k eigenvalues and eigenvectors
def EigDV(covMat, p):
    D, V = np.linalg.eig(covMat) # Get eigenvalues and eigenvectors
    k = Percentage2n(D, p) # Determine k value
    #print("Retain 99%Information, the number of features after dimensionality reduction:"+str(k)+"\n")
    print("Retain 90%Information, the number of features after dimensionality reduction:"+str(k)+"\n")
    eigenvalue = np.argsort(D)
    K_eigenValue = eigenvalue[-1:-(k+1):-1]
    K_eigenVector = V[:,K_eigenValue]
    return K_eigenValue, K_eigenVector
    
#Data after dimensionality reduction
def getlowDataMat(DataMat, K_eigenVector):
    return DataMat * K_eigenVector

#Refactoring data
def Reconstruction(lowDataMat, K_eigenVector, meanVal):
    reconDataMat = lowDataMat * K_eigenVector.T + meanVal
    return reconDataMat

#PCA algorithm
def PCA(data, p):
    dataMat = np.float32(np.mat(data))
    #Data Centralization
    dataMat, meanVal = Z_centered(dataMat)
    #Computation of covariance matrix
        #covMat = Cov(dataMat)
    covMat = np.cov(dataMat, rowvar=0)
    #Get the largest k eigenvalues and eigenvectors
    D, V = EigDV(covMat, p)
    #Data after dimensionality reduction
    lowDataMat = getlowDataMat(dataMat, V)

    #Show PCA 
    #plt.scatter(lowDataMat,[0] * len(lowDataMat))
    #plt.axis('equal')
    #plt.show()
    print('here')
    #Refactoring data
    reconDataMat = Reconstruction(lowDataMat, V, meanVal)
    return reconDataMat

def main():
    #imagePath = 'D:/desktop/banana.jpg'
    #image = cv.imread(imagePath)
    #image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    #rows,cols=image.shape
    #print("The number of features before dimensionality reduction:"+str(cols)+"\n")
    #print(image)
    #print('----------------------------------------')
    #reconImage = PCA(image, 0.99)
    global X_raw, X
    X_raw = X()
    X = PCA(X_raw, 0.90)
    #reconImage = reconImage.astype(np.uint8)
    #print(reconImage)
    #cv.imshow('test',reconImage)
    #cv.waitKey(0)
    #cv.destroyAllWindows()


if __name__=='__main__':
    main()
