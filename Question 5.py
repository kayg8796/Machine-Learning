import numpy as np
import matplotlib.pyplot as plt
from random import uniform

#generating gaussina noise
var_n = 0.05
theta_variance = (0.1 , 2)
id = np.identity(6)

sample_size = (20,500)


#that creates the list of x coordinates for the training set points
theta_true = [0.2,-1,0.9,0.7,0,-0.2] 
theta = [-0.004,-10.54,0.465,0.0087,-0.093,0] 


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

xtest = []
for i in range(20):
    xtest.append(uniform(0,2))

for n in sample_size:
    for var_t in theta_variance:
        x=np.linspace(0,2,n)   
        xtrain = inputMatrix(x)
        ytrain_real= (np.dot(xtrain,theta_true)).tolist()
        #obtaining the y output
        noise =np.random.normal(0,np.sqrt(var_n),n)
        ytrain = np.add(np.dot(xtrain,theta_true), noise)  # this is the y
        
        xTx = np.dot(np.transpose(xtrain),xtrain)
        #using the mean for full bayesian inference
        bracket_term = np.add(((1/var_t)*id),(1/var_n)*xTx)
        term_2 = np.dot(np.transpose(xtrain), np.subtract(ytrain,np.dot(xtrain,theta)))
        term_3=(1/var_n)*( np.dot(np.linalg.inv(bracket_term) , term_2))
        
        theta_pred = np.add(theta,term_3)
        #ytrain_pred = np.dot(xtrain, theta_pred)
        #variance_matrix = np.linalg.inv(bracket_term)
          
            
        ytest_pred=np.dot(inputMatrix(xtest),theta_pred)
        
        y_mean_pred=[]
        for i in range(20):
            out_term = var_n * var_t * inputMatrix(xtest)[i]
            b_term = np.linalg.inv(np.add((var_n * id),(var_t * xTx)))
            y_mean_pred.append(var_n + np.dot(np.dot(out_term,b_term),np.transpose(inputMatrix(xtest)[i])))
        
        plt.errorbar(xtest,ytest_pred,yerr=y_mean_pred, label = 'test set prediction with variance ',capsize = 3,fmt = '.g')
        plt.plot(x,ytrain_real,label = 'true model')
        plt.legend()
        plt.xlabel('x-values')
        plt.ylabel('y-values')
        plt.title('noise variance ={},sample size={} and theta_variance = {}'.format(var_n,n,var_t))
        plt.show()
