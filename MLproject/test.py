import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pandas import DataFrame

data=pd.read_csv("/home/karthik/Desktop/MLproject/utavg.csv",header=None,skiprows=1)


df=DataFrame(data)
x=df.iloc[:, 0:3]
y=df.iloc[:, 3:4]
diff=pd.DataFrame(df.iloc[:, 4:5])


linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(x, y)  # perform linear regression
Y_pred = linear_regressor.predict(x)  # make predictions

plt.scatter(diff,y)
plt.plot(diff,Y_pred,'red')
plt.show()
accuracy = linear_regressor.score(x,Y_pred)
print(accuracy*100,'%')