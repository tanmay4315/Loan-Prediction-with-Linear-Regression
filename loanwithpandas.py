import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from math import sqrt
path="loanTrain.txt"
data=pd.read_csv(path)
labels=list(data['Loan_Status'])
data.pop('Loan_Status')
#data Cleaning
data['Gender'].fillna('Male',inplace=True)
data['Married'].fillna('Yes',inplace=True)
data['Dependents'].fillna('0',inplace=True)
data['Self_Employed'].fillna('No',inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].mean(),inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(),inplace=True)
data['Credit_History'].fillna(1.0,inplace=True)
data.pop('Loan_ID')
nrows=len(data)
columnname=list(data.columns)
for col in columnname:
    featureset=list(set(data[col]))
    for i in range(0,nrows):
        key=data.loc[i,col]
        if data[col].dtype == 'object':
            ind=featureset.index(key)
            data.loc[i,col]=ind
        else:
            break
xtrain=[]
ytrain=[]
xtest=[]
ytest=[]
for i in range(nrows):
    if i%3!=0:
        xtrain.append(data.loc[i])
    else:
        xtest.append(data.loc[i])
x_train=np.array(xtrain)
x_test=np.array(xtest)
nrows=len(x_train)
ncols=len(x_train[0])
setlabel=list(set(labels))
for i in range(0,len(labels)):
    ind=setlabel.index(labels[i])
    labels[i]=ind
for i in range(len(labels)):
    if i%3!=0:
        ytrain.append(labels[i])
    else:
        ytest.append(labels[i])
y_train=np.array(ytrain)
y_test=np.array(ytest)
#Calculating mean and sd
frows=len(x_train)
xmean=[]
xsd=[]
for i in range(ncols):
    col=[num for num in range(frows)]
    mean=sum(col)/frows
    xmean.append(mean)
    coldiff=[(x_train[j][i]-mean) for j in range(frows)]
    sumsq=sum([coldiff[i]**2 for i in range(frows)])
    stdev=sqrt(sumsq/frows)
    xsd.append(stdev)
beta=[0.0]*ncols
betamat=[]
betamat.append(list(beta))
nsteps=350
stepsize=0.004
#making model
for i in range(nsteps):
    residual=[0.0]*frows
    for j in range(frows):
        labelshat=sum([x_train[j][k]*beta[k] for k in range(ncols)])
        residual[j]=y_train[j]-labelshat
    corr=[0.0]*ncols
    for j in range(ncols):
        corr[j]=sum([x_train[k][j]*residual[k] for k in range(frows)])/frows
    istar=0
    corrstar=corr[0]
    for j in range(1,(ncols)):
        if abs(corrstar)<abs(corr[j]):
            istar=j
            corrstar=corr[j]
    beta[istar]+=stepsize*corrstar/abs(corrstar)
    betamat.append(list(beta))
#Getting accuracy
count=0
for i in range(0,len(x_test)):
    y=sum([x_test[i][j]*beta[j] for j in range(ncols)])
    if y==y_test[i]:
        count+=1
accuracy=(count/len(x_test))*100
print("Accuracy of data:",accuracy)
