#cross check the formular here
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
k=5
n=500
id = np.identity(6)
#generating gaussina noise
variance = 0.05
np.random.seed(0)
noise =np.random.normal(0,np.sqrt(variance),n)

x= np.linspace(0,2,n)
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
x_T = np.transpose(xtrain)
xTx= np.dot(x_T,xtrain)
ytrain = np.add(np.dot(xtrain,theta), noise )
y=ytrain
y_true = np.dot(xtrain,theta)
#initializing the required variables
a , b ,var_ty, mean_ty ,A,B =[] , [] , [] , [],[],[]

#initialising second values for a and b so as to aid in the comparison of the while loop
a.append(1)
b.append(1)
i=0
e=0.00000001
while True :
    var_ty.append(np.linalg.inv(np.add(a[i]*id, b[i]*xTx)))
    mean_ty.append(b[i] * np.dot(var_ty[i],np.dot(x_T,y)))
    A.append(np.dot(np.transpose(mean_ty[i]),mean_ty[i]) + np.matrix.trace(var_ty[i]))
    B.append(np.dot( np.transpose(np.subtract(y,np.dot(xtrain,mean_ty[i])))  , np.subtract(y,np.dot(xtrain,mean_ty[i]))) + np.matrix.trace(np.dot(xtrain,np.dot(var_ty[i],x_T))))
    a.append(k/A[i])
    b.append(n/B[i])
    if abs(a[i+1] - a[i])<e and abs(b[i+1] - b[i])<e:
        break
    i += 1
    
    
    
#the estimate is simply the mean 
xtest = []
for j in range(20):
    xtest.append(uniform(0,2))
theta_pred = mean_ty[len(mean_ty) - 1]
y_pred = np.dot(inputMatrix(xtest),theta_pred)
noise_var = [1/t for t in b]
theta_var = [1/v for v in a]
var_n = noise_var[len(noise_var)-1]
var_t = theta_var[len(theta_var)-1]

err=[]
for u in range(20):
    out_term = var_n * var_t * inputMatrix(xtest)[u]
    b_term = np.linalg.inv(np.add((var_n * id),(var_t * xTx)))
    err.append(var_n + np.dot(np.dot(out_term,b_term),np.transpose(inputMatrix(xtest)[u])))


plt.errorbar(xtest,y_pred,yerr=err,capsize = 3,fmt = '.g', label='predicted points with variance')
plt.plot(x,y_true,label = 'true model')
plt.legend()
plt.xlabel('x-values')
plt.ylabel('y-values')
plt.title('Expectation Maximization Prediction results with {} iterations and convergence \n sensitivity : {}'.format(i,e))
plt.show()

noise_var.pop(0)
theta_var.pop(0)
iterations = (np.linspace(1,i+1,i+1)).tolist()
plt.plot(iterations,len(iterations)*[variance],label='actual noise variance')
plt.plot(iterations,noise_var,label='convergence curve')
plt.legend()
plt.ylabel('noise variance')
plt.xlabel('iterations')
plt.title('noise-variance vs number of iterations')
plt.show()

print('the noise variance found is : {}\n'.format(var_n))

