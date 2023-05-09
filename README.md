UW-EE399-Assignment-4
=========
This holds the code and backing for the fourth assignment of the EE399 class. The main dataset this assignment revolves around is the MNIST 784 which is a set of 70000 28x28 images of digits. These digits are processed in order to be categorized by the PCA method. After the dimensions have been reduced, the data is fit with a FFNN a LSTM a SVM and a Decision Tree model in order to compare their effectiveness. 

Project Author: Elijah Reeb, elireeb@uw.edu

.. contents:: Table of Contents

Homework 4
---------------------
Introduction
^^^^^^^^^^^^
This assignment involves two main parts. The first involves fitting a neural network to the data from Homework Assignment 1 (see https://github.com/ElijahReeb/UW-EE399-Assignment-1). These 30 points were split in order to just compare a neural network to a simple linear regression model. 

The second part involves the MNIST dataset. After a 20 component Principle Component Analysis (PCA) the MNIST data is put into a Feed Forward Neural Network (FFNN) a Long Short-Term Memory Network (LSTM) as well as a SVM and Decision tree classifier, similar to homework 3. These models are then compared. 

Theoretical Backgroud
^^^^^^^^^^^^
Feed Forward Neural Networks are made up of a series of layers of neurons. These layers are connected by weights and activation functions. The weights are what are changed during training and lead an input to an output. This is shown in the image below from brilliant.org:

.. image:: ![image](https://user-images.githubusercontent.com/130190276/236995699-30877266-0ba2-422b-9ca8-5fd4ba1c56e4.png)

We can observe the input layer is some amount coming from the format of the input. This may be the pixels of an image or in our case the 20 PCA components of each image. Then the data is passed to middle "hidden layers" where it is processed. Finally, these middle layers converge to an output layer. This output layer has the amount of outputs as one expects or wants categorized to. In the MNIST case this is easy because there are 10 different digits as outputs. The output neuron that is largest is what the model decides the output is for an image. 

Theoretically, there are infinite combinations of layers and layer sizes as well as activation functions. Through some simple hyper-parameter tuning this model just uses a 3 layer network to classify. Through testing and comparing errors one can determine an acceptable combination with minimal error. However, this is largely unknown at the start. 

Because of being the main focus of Assignment 3 the theoretical background of SVM and Decision Tree classifiers will not be explained here. Additionally, LSTM will be covered in a later assignment. 

Algorithm Implementation and Development
^^^^^^^^^^^^
With the ease of pytorch packages, FFNN code is simple to develop. After importing the necessary tools 4 main functions are called in order to setup train and test a neural network. They are broken up below. The first block involves setting the parameters of the layers and their size. For part I of this assignment a simple neural net is used with a 1 dimentional input and output. Those are the first ones. And the middle layers of 20 and 10 are set up next. These values are again from hyper-parameter tuning, but can be a large range. 

.. code-block:: text

        class Net(nn.Module):
                def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 20)
                self.fc2 = nn.Linear(20, 10)
                self.fc3 = nn.Linear(10, 1)

Next, the layers are connected in the feed forward method with set activation functions. In this case the "relu" activation function is used which is 0 for x less than 0 and x for x > 0. The goal of this actvation function is to be easily differentiable because when taking the derivatives via chainrule to implement the backpropigation of weights, it saves a lot of computing resources to have easily differentiable functions. 

.. code-block:: text

         def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x

After the model infrastructure is set up, the model is trained. A few commands at the top define the type of loss function that will be used. In this case MSE (mean square error) will be used. Additionally, the optimizer is chosen. We will be using SGD (stochastic gradient descent) which involves randomly selecting groups of points and computing the gradient for those points before "stepping down the mountain" instead of taking the gradient of all points. In the simple model this is not even really necessary, but it is very crucial when looking at MNIST data. Additionally, a learning rate is defined for how large of steps will be taken. This is 0.05 for now but is another hyperparameter to be changed with testing. 

.. code-block:: text

        net = Net()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

        for epoch in range(1000):
            optimizer.zero_grad()  # zero the gradient buffers
            outputs_pred = net(inputs)  # forward pass
            loss = criterion(outputs_pred, outputs)  # calculate the loss
            loss.backward()  # backward pass
            optimizer.step()  # update the weights

Finally, 

.. code-block:: text

        with torch.no_grad():
            outputs_pred = net(testinputs)
            loss = criterion(outputs_pred, testoutputs)
            train_pred = net(inputs)
            trainloss = criterion(train_pred, outputs)
            print(f"Loss: {loss}")
            print(f"Training Loss: {loss}")

These simple code blocks allow one to create and train a neural net on a set of data. 


Computational Results
^^^^^^^^^^^^


Summary and Conclusions
^^^^^^^^^^^^
