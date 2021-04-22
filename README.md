# EECS_738_project3
Says One Neuron To Another

## data to present
1.Statlog(Heart) Data Set

2.

## method
To demonstrate the neural network process, three main layers are to be conducted.

![1_NZc0TcMCzpgVZXvUdEkqvA](https://user-images.githubusercontent.com/42806161/115626377-0ae47980-a2c3-11eb-9be8-eb251436308b.png)

In general, we would design a network as shown above with an input layer, several hidden layers, and a output layer, where  

between each layers activation functions were used as a threshold.

* data preperation

 **readcsv**  and **pandas** are used for data sorting. For seperating test/train data sections, sklearn **train_test_split** is 
 
 used to make life better. below  is the header printed with first several rows of data. As shown in this data we had 13 features.
 
 before going further in reallife, data must be checked if there's missing values and what the data type is for each feature.
 
![image](https://user-images.githubusercontent.com/42806161/115646609-1a29ee00-a2e8-11eb-8950-f5fa7d5ec8ff.png)

the data is splited as shown:

![image](https://user-images.githubusercontent.com/42806161/115647520-89541200-a2e9-11eb-915c-69a5c8ba5835.png)

* design the hiden layer(s)

In this heart data, one hidden layers are specified with 13 nodes which needs to match the input layer.

**weights** and **bias** are initialized by given random numbers from a random normal distribution with ragard to the size of input/hidden/output layer and stored in a dictionary **params**

Note that we also set the learning rate which basically defines when we want the training to stop and iterations of the training.

* activation functions

In this neural network, I played with both sigmoid and ReLu activation function to get familiar with them. But the rationale of 
using sigmoid at the output layer is we have a binary class output.

* loss function

A loss function is also specify in the study for backward propagation when we want to update the weights and bias. The choice of loss function is dependent on tasks, in our case here, we are trying to do classification problems where cross-entropy loss is preferred. 

![1_e8qhWEz_8PZBLmF37JHfSA](https://user-images.githubusercontent.com/42806161/115663263-e231a380-a305-11eb-9943-3636ac71294f.png)

Furthermore, for binary classification task as the heart data here, the following loss function is used:

![image](https://user-images.githubusercontent.com/42806161/115663471-2b81f300-a306-11eb-9a4a-5e3782a85116.png)

* forward propagation

In this phase we need to compute the weighted sum between input and first layer with 'Z1 = (W1 * X) + b', then pass the result **Z1** to our activation function(ReLu) and get A1. With that, the second weighted sum would be computed based on the previous result A1, which is writen 'Z2 = (W2 * A1) + b2', then **Z2** will be pass into the output layer's activation function returning A2. Finally, the loss during propagation should be computed between A2 and true label.

* backward propagation

In order to minimize the loss here, We do this by taking the derivative of the error function, with respect to the weights (W) of our NN, using gradient descent. As shown in the code, derivative of loss function in respect to output **yhat** is calculated, then passing backward to hidden layer output Z2, Z1. The derivative of all the variables involved are then computed, except the input X.

then that's where we can update our weights and bias with the loss with respect to each variables by deducting them with the 'learning rate' and 'loss': **self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1**

## results and discussion
For the first testing data, I chose the heartbeat data to start with, since it has a straight forawrd binary outcome. 

![image](https://user-images.githubusercontent.com/42806161/115666008-a26cbb00-a309-11eb-92cf-7212aea4173b.png)

Given learning rate=0.01 and interation=1000, the loss curve is as shown. which indicated the training goes smoothly.

I then test the neural network by feeding test data, the results is at aboout 74% acccuracy which is not bad.

![image](https://user-images.githubusercontent.com/42806161/115666283-fe374400-a309-11eb-9da8-b8c950c326f6.png)

So far, the basic 2 layer neural network is successfully built and implemented.
