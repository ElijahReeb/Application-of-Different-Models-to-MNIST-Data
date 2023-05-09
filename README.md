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
The second part involves the MNIST dataset. After a 20 component Principle Component Analysis (PCA) 

Theoretical Backgroud
^^^^^^^^^^^^

.. image:: 


Algorithm Implementation and Development
^^^^^^^^^^^^
With the ease of pytorch packages, FFNN code is simple to develop. After importing the necessary tools 4 main functions are called in order to 

.. code-block:: text

        class Net(nn.Module):
                def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 20)
                self.fc2 = nn.Linear(20, 10)
                self.fc3 = nn.Linear(10, 1)

Next 

.. code-block:: text

         def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x

After this 

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
