#!/usr/bin/env python
# coding: utf-8

# Breast Cancer Wisconsin data set is used to predict whether a cancer is benign or malignant. A 2 layer Neural Network is trained and tested.

# In[123]:


import numpy as np
import pandas as pd
import csv
import numpy as np
import warnings
warnings.filterwarnings("ignore") #suppress warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# headers =  ['age', 'sex','chest_pain','resting_blood_pressure',  
#         'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
#         'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',"slope of the peak",
#         'num_of_major_vessels','thal', 'heart_disease']
Breast_df = pd.read_csv('data.csv')#, sep=' ', names=headers


# In[124]:


#show data
Breast_df.describe()
#heart_df.shape


# data prep

# classifier and label columns need to be dropped and replace diagnosis label to 0/1 class: M (malignant) = 1 and B (Benign) = 0

# In[125]:


#binary class data case
#convert input as np.array
X = Breast_df.drop(columns=['id','Unnamed: 32'], axis = 1)
X.head()


# replace target class with 0 and 1 for binary classes

# In[134]:


#replace target class with 0 and 1 
#1 means "have heart disease" and 0 means "do not have heart disease"
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(X['diagnosis'])
X['diagnosis'] = le.transform(X['diagnosis'])
X.head()


# normalize/standarize data

# In[132]:


X_norm = Breast_df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
X_norm = pd.concat([X_norm, X['diagnosis']], axis = 1)
X_norm.describe()


# In[148]:


XX = X.drop(columns=['diagnosis'])
y_label = X_norm['diagnosis'].values.reshape(XX.shape[0], 1)
Xtrain, Xtest, ytrain, ytest = train_test_split(XX, y_label, test_size=0.2, random_state=2)
sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)
Xtrain


# In[145]:


y_label = X_norm['diagnosis'].values.reshape(X.shape[0], 1)

#split data into train and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(X_norm, y_label, test_size=0.2, random_state=2)

# #standardize the dataset
# sc = StandardScaler()
# sc.fit(Xtrain)
# Xtrain = sc.transform(Xtrain)
# Xtest = sc.transform(Xtest)
# transfer to np.array 
Xtrain = Xtrain.to_numpy()
Xtest = Xtest.to_numpy()
Xtrain


# In[149]:


print(f"Shape of train set is {Xtrain.shape}")
print(f"Shape of test set is {Xtest.shape}")
print(f"Shape of train label is {ytrain.shape}")
print(f"Shape of test labels is {ytest.shape}")


# NN structure

# In[150]:


class NN():
    def __init__(self, layers=[31,20,1], learning_rate=0.001, iterations=1000):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.y = None

    def init_weights(self):
        np.random.seed(1) # Seed the random number generator
        self.params["W1"] = np.random.randn(self.layers[0],self.layers[1]) 
        self.params['b1'] = np.random.randn(self.layers[1],)
        self.params['W2'] = np.random.randn(self.layers[1],self.layers[2]) 
        self.params['b2'] = np.random.randn(self.layers[2],)

    def eta(self, x):
        ETA = 0.0000000001
        return np.maximum(x, ETA)

    def entropy_loss(self,y, yhat):
        nsample = len(y)
        yhat_inv = 1.0 - yhat
        y_inv = 1.0 - y
        yhat = self.eta(yhat) ## clips value to avoid NaNs in log
        yhat_inv = self.eta(yhat_inv) 
        loss = -1/nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((y_inv), np.log(yhat_inv))))
        return loss
  
    def ReLu(self,Z):
        return np.maximum(0.0, Z)

    def dReLu(self,x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    #activation function and derievative
    def sigmoid(self,Z):
        '''
        The sigmoid function takes in real numbers in any range and 
        squashes it to a real-valued output between 0 and 1.
        '''
        return 1/(1+np.exp(-Z))

    def dsigmoid(self,Z):
        sig = sigmoid(Z);
        return sig * (1.0 - sig); 

    def forward(self):
    # self.layer = ReLu(np.dot(self.input, self.weights1))
    # self.output = ReLu(np.dot(self.layer, self.weights2))
        Z1 = self.X.dot(self.params['W1']) + self.params['b1']
        A1 = self.ReLu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        yhat = self.sigmoid(Z2)
        loss = self.entropy_loss(self.y,yhat)

        # save calculated parameters in dictionary    
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return yhat,loss

    def backward(self, yhat):
        '''
        Computes the derivatives and update weights and bias according.
        '''
        y_inv = 1 - self.y
        yhat_inv = 1 - yhat

        # the loss with respect to
        dl_wrt_yhat = np.divide(y_inv, self.eta(yhat_inv)) - np.divide(self.y, self.eta(yhat))
        dl_wrt_sig = yhat * (yhat_inv)
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)

        dl_wrt_z1 = dl_wrt_A1 * self.dReLu(self.params['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)

        #update the weights and bias
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2

    def fit(self, X, y):
        '''
        Trains the neural network using the specified data and labels in iteration
        '''
        self.X = X
        self.y = y
        self.init_weights() #initialize weights and bias

        for i in range(self.iterations):
            yhat, loss = self.forward()
            self.backward(yhat)
            self.loss.append(loss)

    def predict(self, X):
        '''
        Predicts on a test data
        '''
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.ReLu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.sigmoid(Z2)
        return np.round(pred)  

    def cal_acc(self,y,yhat):
        '''
        Calculates the accutacy between the predicted valuea and the truth labels
        '''
        acc = int(sum(y == yhat) / len(y) * 100)
        return acc    

    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()  


# **loss functions:**
# 
# A loss function must be properly designed so that it can correctly penalize a model that is wrong and reward a model that is right.
# 
# One of the simplest loss functions used in deep learning is MSE, or mean square error
# 
# ```
# def nloss(self,Yh):
#         loss = (1./self.sam) * (-np.dot(self.Y,np.log(Yh).T) - np.dot(1-self.Y, np.log(1-Yh).T))    
#         return loss
# ```
# for classification problems, you can use cross-entropy loss
# 
# 
# 
# ```
# def eta(self, x):
#   ETA = 0.0000000001
#   return np.maximum(x, ETA)
# 
# def entropy_loss(self,y, yhat):
#     nsample = len(y)
#     yhat_inv = 1.0 - yhat
#     y_inv = 1.0 - y
#     yhat = self.eta(yhat) ## clips value to avoid NaNs in log
#     yhat_inv = self.eta(yhat_inv) 
#     loss = -1/nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((y_inv), np.log(yhat_inv))))
#     return loss
# ```
# 
# 
# 

# demo on heart data

# training phase

# In[154]:


nn = NN(layers=[30,15,1], learning_rate=0.01, iterations=1000) #create NN model
nn.fit(Xtrain, ytrain)
nn.plot_loss()


# testing phase

# In[155]:


train_pred = nn.predict(Xtrain)
test_pred = nn.predict(Xtest)

print("Train accuracy is {} %".format(nn.cal_acc(ytrain, train_pred)))
print("Test accuracy is {} %".format(nn.cal_acc(ytest, test_pred)))


# sklearn cheat sheet result

# In[156]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

sknet = MLPClassifier(hidden_layer_sizes=(8), learning_rate_init=0.001, max_iter=100)
sknet.fit(Xtrain, ytrain)
preds_train = sknet.predict(Xtrain)
preds_test = sknet.predict(Xtest)

print("Train accuracy of sklearn neural network: {}".format(round(accuracy_score(preds_train, ytrain),2)*100))
print("Test accuracy of sklearn neural network: {}".format(round(accuracy_score(preds_test, ytest),2)*100))


# 
