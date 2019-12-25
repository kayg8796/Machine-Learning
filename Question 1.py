import numpy as np
import matplotlib.pyplot as plt
import random
n=20
#generating gaussina noise
variance = 0.1
#np.random.seed(4)
def noise(num): 
    return np.random.normal(0,np.sqrt(variance),num)


x=np.linspace(0,2,n)
#that creates the list of x coordinates for the training set points 
theta = np.transpose([-1.0,0.9,0.7,0.0,-0.2,0.2])

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
    xtr = np.c_[x1,x2,x3,x4,x5,oness]
    return xtr
    
xtrain = inputMatrix(x)
ytrain_real = (np.dot(xtrain,theta)).tolist()


#obtaining the y output

ytrain = np.add(np.dot(xtrain,theta), noise(20))
#now the training set is ready 
xtrainT = np.matrix.transpose(xtrain)
xTy = np.dot(xtrainT,ytrain)
xTx = np.dot(xtrainT,xtrain)
xTx_inv = np.linalg.inv(xTx)
theta_pred = np.dot(xTx_inv, xTy)


#calculation of mse over training set data
ytrain_pred = np.dot(xtrain,theta_pred) #taking out the noise , used it just for the generation of the training dataset

def mse(ytrain_real1,ytrain_pred1):
    sum1=0
    for i in range(0,len(ytrain_real1)):
        sum1 += (ytrain_real1[i] - ytrain_pred1[i])**2
    #mse_train = sum1
    return sum1/len(ytrain_real1)
    
mse_training = mse(ytrain_real,ytrain_pred)



#now for the test sample
# just added the noise to the test generation and also change the comparison to the data.
xtest1 = []
for j in range(1000):
    xtest1.append(random.uniform(0,2))
xtest_matrix = inputMatrix(xtest1)
ytest_real = np.dot(xtest_matrix, theta)
ytest_n = np.add(ytest_real,np.random.normal(0,np.sqrt(variance),1000))
ytest_pred =np.dot(xtest_matrix,theta_pred)
ytest_pred_n = np.add(ytest_pred,noise(1000))
mse_test = mse(ytest_real,ytest_pred_n)


plt.plot(x,ytrain_real,label='true model')
plt.plot(x,ytrain_pred,label='predicted model')
plt.legend()
plt.title(' training set')
plt.show()
plt.scatter(xtest1,ytest_real,label='true model')
plt.scatter(xtest1,ytest_pred_n,label='predicted model')
plt.legend()
plt.title('test set performance with noise')
plt.show()

print('the MSE of the training set is : {}\n The MSE of the test set is {}'.format(mse_training,mse_test))


