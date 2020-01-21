import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from webbrowser import open_new_tab
import webbrowser
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data=pd.read_csv("/home/karthik/Desktop/MLproject/dataset.csv",header=None,skiprows=1)


df=DataFrame(data)



#only k1, k2, k3 values
x=df.iloc[:, 0:3]
k1=pd.DataFrame(df.iloc[:, 0:1])
k2=pd.DataFrame(df.iloc[:, 1:2])
k3=pd.DataFrame(df.iloc[:, 2:3])
diff=pd.DataFrame(df.iloc[:, 5:6])

#parameters for gradient descent
m=x.shape[0]
alpha=0.0001
no_iter=1000
bias=np.random.rand(1)

#average marks alone
y=df.iloc[:, 3:4]

theta=np.random.rand(3).reshape(3,1)

def eqn(x,theta,bias):
	ans=np.dot(x,theta)+bias
	#print(ans)
	return ans

def cost_func(m,y,pred):
	squared_error=np.square(pred-y)
	ans=np.sum(squared_error)/(2*m)
	#print(ans)
	return np.array(ans)[0]

def differential(m,y,pred,x):
	ans=(np.sum(np.dot(np.transpose(pred-y),x)))/m
	#print(ans)
	return ans


def gradient_descent(m,x,y,theta,no_iter,alpha):
	ind=[]
	cost=[]
	for i in range(no_iter):
		pred=eqn(x,theta,bias)
		pred.reshape(m,1)
		gradient0=differential(m,y,pred,k1)
		gradient1=differential(m,y,pred,k2)
		gradient2=differential(m,y,pred,k3)
		theta[0][0]=theta[0][0]-(alpha*gradient0)
		theta[1][0]=theta[1][0]-(alpha*gradient1)
		theta[2][0]=theta[2][0]-(alpha*gradient2)
		print(theta,cost_func(m,y,pred))
	return theta

theta=gradient_descent(m,x,y,theta,no_iter,alpha)

output=np.dot(x,theta)
print(theta)

print(np.sqrt(metrics.mean_squared_error(y,output)))
plt.scatter(diff,y)
plt.xlabel('difficulty')
plt.ylabel('average marks')

#plt.plot(diff,output,'r')
plt.scatter(diff,output)

plt.show()



