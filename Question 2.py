import numpy as np
import matplotlib.pyplot as plt
n=20 #number of samples

variance = 0.1
def noise(var,num): #function to generate noise arrays each time it is called for the different algorithms
    return np.array(np.random.normal(0,np.sqrt(var),num))

x=np.linspace(0.1,2,n) #generation of n samples between 0 and 2 with intervals of 2/n

theta = [0.2,-1,0.9,0.7,0,-0.2]  #true theta given by the question

#function to generate input matrix for the 5th generation polynomial inorder to use for the generating the training set
def fifth_degree_polynomial(x1): #true model which is used to generate the training dataset
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
xtrain_second_deg_fifth_deg = fifth_degree_polynomial(x)
ytrain_real =(np.dot(xtrain_second_deg_fifth_deg,theta)).tolist() #true value of y

    
# generating of different y training sets for the 100 experiments using different noise samples
ytrain=[]
for i in range(100): 
    ytrain.append((np.dot(xtrain_second_deg_fifth_deg,theta) + noise(variance,n)).tolist()) 

#function to generate input matrix for the second degree polynomial which would be used to predict the system model
def second_degree_polynomial(x1):
    oness=np.ones(int(len(x1)))
    x2=[]
    for i in range(0,len(x1)):
        x2.append(x1[i]**2)
    xtr = np.c_[oness,x1,x2]
    return xtr

    
#generating xtrain_second_deg for second degree polynomial assuming that it was a second deg that was used to produce the ytrain data
xtrain_second_deg = second_degree_polynomial(x) 


#now the training set is ready 
xtrain_second_degT = np.matrix.transpose(xtrain_second_deg)
xTx = np.dot(xtrain_second_degT,xtrain_second_deg)
xTx_inv = np.linalg.inv(xTx)

#here we predict different theta values for the 100 different experiments carried out producing different y values
theta_pred=[]
xTy=[]
for i in range(100):
    xTy.append( np.dot(xtrain_second_degT,ytrain[i]))
    theta_pred.append(np.dot(xTx_inv, xTy[i]))


#we use the different theta values to predict different y values  and then calculate the mean and variance
ytrain_pred_deg2=[]
for i in range(0,100):
    ytrain_pred_deg2.append(np.dot(xtrain_second_deg,theta_pred[i])) 
sum2=[]   
variance_deg_2 = []
sparse = np.array(ytrain_pred_deg2)
for j in range(0,20):
    sum2.append(np.sum(sparse[:,[j]]))
    
    
ymean_deg_2 = [k/100 for k in sum2]  
sum1=[]
for i in range(0,20):
    sum11=0
    for j in range(1,100):
        sum11 += ( pow(sparse[j][i] - ymean_deg_2[i], 2))
    sum1.append(sum11)
variance_deg_2 = [k/100 for k in sum1]
    
    
    

plt.plot(x,ytrain_real,label = 'true model')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.plot(x,ymean_deg_2,label = 'mean value of predictions')
plt.title('Graph showing true model generated using 5th degree polynomial and the \n\
          the mean of 100 predicted samples using  2nd degree polynomial')
plt.legend()
plt.show()
#plt.scatter(x,variance_deg_2)
#plt.show()

for i in range(0,len(sparse[0])):
    plt.plot(x,sparse[i])
plt.plot(x,ytrain_real)
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.title('Graph showing true model generated using 5th degree polynomial and the \n\
           100 predicted samples using  2nd degree polynomial')
plt.show()


def tenth_degree_polynomial(x1):
    oness=np.ones(int(len(x1)))
    x2 , x3,x4,x5,x6,x7,x8,x9,x10 = [] , [], [],[] , [], [],[] , [], []
    for i in range(0,len(x1)):
        x2.append(x1[i]**2)
        x3.append(x1[i]**3)
        x4.append(x1[i]**4)
        x5.append(x1[i]**5)
        x6.append(x1[i]**6)
        x7.append(x1[i]**7)
        x8.append(x1[i]**8)
        x9.append(x1[i]**9)
        x10.append(x1[i]**10)
    xtr = np.c_[oness,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]
    return xtr

xtrain_tenth_deg = tenth_degree_polynomial(x)

#now the training set is ready 
xtrain_tenth_degT = np.matrix.transpose(xtrain_tenth_deg)
xTx_10 = np.dot(xtrain_tenth_degT,xtrain_tenth_deg)
xTx_inv_10 = np.linalg.inv(xTx_10)

#here we predict different theta values for the 100 different experiments carried out producing different y values
theta_pred_10=[]
xTy_10=[]
for i in range(100):
    xTy_10.append( np.dot(xtrain_tenth_degT,ytrain[i]))
    theta_pred_10.append(np.dot(xTx_inv_10, xTy_10[i]))
    
#we use the different theta values to predict different y values  and then calculate the mean and variance
ytrain_pred_deg10=[]
for i in range(0,100):
    ytrain_pred_deg10.append(np.dot(xtrain_tenth_deg,theta_pred_10[i])) 
sum10=[]   
variance_deg_10 = []
sparse10 = np.array(ytrain_pred_deg10)
for j in range(0,20):
    sum10.append(np.sum(sparse10[:,[j]]))
    
ymean_deg_10 = [k/100 for k in sum10]  
sum110=[]
for i in range(0,20):
    sum11=0
    for j in range(1,100):
        sum11 += ( pow(sparse10[j][i] - ymean_deg_10[i], 2))
    sum110.append(sum11)
variance_deg_10 = [k/100 for k in sum110]


plt.plot(x,ytrain_real,label = 'true model')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.plot(x,ymean_deg_10,label = 'mean value of predictions')
plt.legend()
plt.title('Graph showing true model generated using 5th degree polynomial and the \n\
          the mean of 100 predicted samples using  10th degree polynomial')
plt.show()
plt.errorbar(x,ymean_deg_2, yerr = variance_deg_2,capsize = 3 , fmt ='.r',label = '2nd degree model with variance')
plt.errorbar(x,ymean_deg_10, yerr = variance_deg_10,capsize = 3 , fmt ='.g',label = '10th degree model with variance')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.plot(x,ytrain_real,label = 'true model')
plt.legend()
plt.title('Comparison between the results ot the 10th degree and \n second degree models')

#plt.scatter(x,variance_deg_10)
plt.show()

for i in range(0,len(sparse10[0])):
    plt.plot(x,sparse10[i])
#plt.plot(x,ytrain_real)
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.title('Graph showing true model generated using 5th degree polynomial and the \n\
           100 predicted samples using  10th degree polynomial')
plt.show()