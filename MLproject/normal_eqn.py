import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from webbrowser import open_new_tab
import webbrowser
from sklearn import metrics


data=pd.read_csv("/home/karthik/Desktop/junk/utavg.csv",header=None,skiprows=1)


df=DataFrame(data)



#only k1, k2, k3 values
x=df.iloc[:, 0:3]


#average marks alone
y=df.iloc[:, 3:4]


k1=pd.DataFrame(df.iloc[:, 0:1])
k2=pd.DataFrame(df.iloc[:, 1:2])
k3=pd.DataFrame(df.iloc[:, 2:3])
diff=pd.DataFrame(df.iloc[:, 4:5])


xinv=pd.DataFrame(np.linalg.pinv(x.values), x.columns, x.index)



theta=pd.DataFrame(np.dot(xinv,y))
print(theta)


output=pd.DataFrame(np.dot(x,theta))

print(np.sqrt(metrics.mean_squared_error(y,output)))


plt.scatter(diff,y)
plt.xlabel('difficulty')
plt.ylabel('average marks')

#plt.plot(diff,output,'r')
plt.scatter(diff,output)

plt.show()

#getting the python output to an html page

sum=0
while(1):
	print('enter the mark distribution')
	ip=[]
	for i in range(3):
		ip.append(int(input()))


	for x in ip:
		sum=sum+x
	if sum>50:
		print("invalid mark distribution")
		sum=0
	else:
		break

op=np.dot(ip, theta)

print('the expected average is ',op[0])

filename='sample.html'
f=open(filename,'w')

wrapper="""<html>
<head></head>
<body>the average mark expected for mark distribution 
k1:%d, k2:%d, k3:%d is <b>AVERAGE:%d<b>
</body>
</html>"""

html_string=wrapper%(ip[0], ip[1], ip[2], op[0])

f.write(html_string)
f.close()
webbrowser.open(filename, 0)
