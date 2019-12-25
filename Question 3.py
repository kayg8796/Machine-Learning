import numpy as np
import matplotlib.pyplot as plt
import random

lamdda = []

n=20
#generating gaussina noise
variance = 0.1
def mse(ytrain_real1,ytrain_pred1):
    sum=0
    for i in range(0,len(ytrain_real1)):
        sum += (ytrain_real1[i] - ytrain_pred1[i])**2
    mse_train = sum/len(ytrain_real1)
    return mse_train
def noise(num):
    return np.random.normal(0,np.sqrt(variance),num)

xvals=(np.arange(0,2,2/float(n))).tolist()
#that creates the list of x coordinates for the training set points 
x=[]

for i in range (0,len(xvals)):
    x.append(xvals[i])
theta = [0.2,-1,0.9,0.7,0,-0.2] 

#function to generate input matrix
def inputMatrix(x1):
    oness=np.ones(int(len(x1)))
    x2=[]
    x3=[]
    x4=[]
    x5=[]
    for i in range(0,len(x1)):
        x2.append(x1[i]**2)
        x3.append(x1[i]**3)
        x4.append(x1[i]**4)
        x5.append(x1[i]**5)
    xtr = np.c_[oness,x1,x2,x3,x4,x5]
    return xtr
    
xtrain = inputMatrix(x)
ytrain = (np.dot(xtrain,theta)).tolist()
ytrain_real = ytrain

#random.seed(0)
xtest1 = [k/1000 for k in random.sample(range(0,2000),1000)]
xtest_matrix = inputMatrix(xtest1)
ytest_real = np.dot(xtest_matrix, theta)

#obtaining the y output
ytrain = np.add(np.dot(xtrain,theta), noise(20))
#now the training set is ready 
xtrainT = np.matrix.transpose(xtrain)
xTy = np.dot(xtrainT,ytrain)
xTx = np.dot(xtrainT,xtrain)
xTx_plus_lambdda , xTx_inv , theta_pred , mse_training , ytrain_pred , ytest_pred , mse_test  = [] ,[] , [] , [] , [] , [] , []

for i in range(0,20):
    k=i/100
    lamdda.append(k)
    xTx_plus_lambdda.append(np.add(xTx,k*(np.identity(6))))
    xTx_inv.append(np.linalg.inv(xTx_plus_lambdda[i]))
    theta_pred.append(np.dot(xTx_inv[i], xTy))
    #calculation of mse over training set data
    ytrain_pred.append(np.dot(xtrain,theta_pred[i])) #taking out the noise , used it just for the generation of the training dataset
    mse_training.append(mse(ytrain_real,ytrain_pred[i]))
    ytest_pred.append(np.add(np.dot(xtest_matrix,theta_pred[i]),noise(1000)))
    mse_test.append( mse(ytest_real,ytest_pred[i]))

plt.scatter(lamdda,mse_training,label = 'training set')
#plt.show()
plt.scatter(lamdda,mse_test,label = 'test set')
#plt.pylab_setup.ylim([0,0.005])
plt.legend(loc='best')
plt.title('MSE vs lambda')
plt.ylabel('MSE')
plt.xlabel('lamda')
plt.show()

