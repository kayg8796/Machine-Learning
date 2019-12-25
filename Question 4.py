#you need to add the variance here

import numpy as np
import matplotlib.pyplot as plt


#generating gaussina noise
variance_noise =( 0.05 , 0.15)
var_t = 0.1
id = np.identity(6)

n = 20

x=np.linspace(0,2,n)
#that creates the list of x coordinates for the training set points 
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

xtest = []
from random import uniform
for i in range(20):
    xtest.append(uniform(0,2))


ytrain = (np.dot(xtrain,theta)).tolist()
ytrain_real = ytrain
#obtaining the y output
for var_n in variance_noise:
    noise =np.random.normal(0,np.sqrt(var_n),n)
    ytrain = np.add(np.dot(xtrain,theta), noise)  # this is the y
    
    xTx = np.dot(np.transpose(xtrain),xtrain)
    #using the mean for full bayesian inference
    bracket_term = np.add(((1/var_t)*id),(1/var_n)*xTx)
    term_2 = np.dot(np.transpose(xtrain), noise)
    term_3=(1/var_n)*( np.dot(np.linalg.inv(bracket_term) , term_2))
    
    theta_pred = np.add(theta,term_3)
    ytrain_pred = np.dot(xtrain, theta_pred)
    variance_matrix = np.linalg.inv(bracket_term)
      
        
    ytest_pred=np.dot(inputMatrix(xtest),theta_pred)
    
    y_mean_pred=[]
    for i in range(20):
        out_term = var_n * var_t * inputMatrix(xtest)[i]
        b_term = np.linalg.inv(np.add((var_n * id),(var_t * xTx)))
        y_mean_pred.append(var_n + np.dot(np.dot(out_term,b_term),np.transpose(inputMatrix(xtest)[i])))
    
    plt.errorbar(xtest,ytest_pred,yerr=y_mean_pred, label = 'test set predictions with variance',capsize = 3,fmt = '.g')
    plt.plot(x,ytrain_real,label='true model')
    plt.legend()
    plt.xlabel('x-values')
    plt.ylabel('y-values')
    plt.title('noise variance ={},sample size={} and theta_variance = {}'.format(var_n,n,var_t))
    plt.show()
