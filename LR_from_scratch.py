#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('F:/archive/creditcard.csv')


# In[3]:


df


# In[4]:


df.dtypes


# In[5]:


# df.describe()


# In[6]:


df['Class'].value_counts()


# In[7]:


def scale(x):
  new=x-np.mean(x,axis=0)
  return new/np.std(x,axis=0)


# In[8]:


df1=df.iloc[:,:-1]


# In[9]:


scaled_x=scale(np.array(df1))


# In[10]:


scaled_xdf=pd.DataFrame(scaled_x)
y=np.array(df['Class'])


# In[11]:


df_yes=df[df['Class']==1]


# In[12]:


df_no=df[df['Class']==0]


# In[13]:


print(df_yes)


# In[14]:


scaled_xdf


# In[15]:


df_class=df.iloc[:,-1]


# In[18]:


# from imblearn.over_sampling import SMOTE
# oversample=SMOTE()
# X, y = oversample.fit_resample(scaled_x,y)

from sklearn.utils import resample
df_downsample = resample(df_no,
             replace=True,
             n_samples=len(df_yes),
             random_state=42)

print(df_downsample.shape)


# In[19]:


data_downsampled = pd.concat([df_downsample, df_yes])


# In[20]:


df3=pd.DataFrame(data_downsampled)
df3['Class'].value_counts()


# In[21]:


df3


# In[22]:


del df


# In[23]:


df3.corr()
y=np.array(df3['Class'])


# In[24]:


df3_noclass=df3.drop(['Class'],axis=1)


# In[25]:


df3_noclass_scale=scale(df3_noclass)


# In[26]:


df3_noclass_scale


# In[27]:


x=np.array(df3_noclass_scale)
print(x.shape)

print(y.shape)


# In[28]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20,test_size=0.12)


# In[29]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[30]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))


# In[31]:


def cost(n,y_true,y_pred):
    return -(1/n)*np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))


# In[32]:


y=np.reshape(y,(-1,1))
print(y.shape)


# In[43]:


def model(x,y,learning_rate,iteration,tol):
    m=x.shape[1]
    n=x.shape[0]
    
    
    count=0
    w=np.zeros((n,1))
    b=0
    cost_list=[]
    while count<=iteration:
#         print(x.shape)
#         print(w.T.shape)
        z=np.dot(w.T,x)+b
        a=sigmoid(z)
#         print("a",a.shape)
#         print("y",y.shape)
#         print("me hu a",a)
        y=np.reshape(y,[-1,1])
#         print("after y ka shape",y.shape)
#         cost=-(1/n)*np.sum( y*np.log(a) + (1-y)*np.log(1-a))
        initial=cost(n,y,a)
#         print("initial,",initial.shape)
        dw=(1/n)*np.dot(a-y,x.T)
        db=(1/n)*np.sum(a-y)

        w_final=w-learning_rate*dw.T
        b_final=b-learning_rate*db
        
        Z=np.dot(w_final.T,x)+b_final
        A=sigmoid(Z)
        
        final=cost(n,y,A)
        
        if np.abs(final-initial)<=tol:
            
            
            break
        
        cost_list.append(final)
        print(f"no of iteration {count}, cost value is {final},")
        count+=1
       
        
            
        w=w_final
        b=b_final
        
    return w,b,cost_list


# In[44]:


w,b,cost_list=model(x_train,y_train,0.01,10000,0.0001)


# In[46]:


import matplotlib.pyplot as plt
b=np.arange(0,10001)
plt.plot(b,cost_list)
plt.show()


# In[48]:


def accuracy(X, Y, W, B):
    
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)
    
    A = A > 0.5
    
    A = np.array(A, dtype = 'int64')
    
    acc = (1 - np.sum(np.absolute(A - Y))/Y.shape[1])*100
    
    print("Accuracy of the model is : ", round(acc, 2), "%")


# In[51]:


x_test=np.reshape(x_test,(1,-1))
y_test=np.reshape(y_test,(1,-1))


# In[ ]:





# In[54]:


print(w.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




