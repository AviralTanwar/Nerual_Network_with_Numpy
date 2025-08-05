# Importing the numpy library for numerical operations
import numpy as np

# Importing the pandas library for data manipulation and analysis
import pandas as pd

# Importing pyplot from matplotlib for data visualization
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------------------------------------------
def init_params():
    """ 
    Here we have initialized the parameters for our model.
    W1 and b1 are the weights and biases for the first layer,
    W2 and b2 are the weights and biases for the second layer.

    We have use random with randn to have the values from -0.5 to +0.5.

    Note -> We Use rand if we want the values between 0,1
    
    Also the dimensions of the layers are as follows:
        Input layer: 784 features (pixels)
        Hidden layer: 10 neurons (can vary, chosen arbitrarily for simplicity)
        Output layer: 10 neurons (one for each digit)

    And THUS the values 10, 784 and so on.
    """
    try:
        W1 = np.random.randn(10, 784)
        b1 = np.random.randn(10, 1)
        W2 = np.random.randn(10, 10)
        b2 = np.random.randn(10, 1)

        return W1, b1, W2, b2
    except Exception as e:
        print("Error in init_params:", e)
        raise

def ReLu(Z):
    """
    ReLU activation function.
    It returns the input directly if it is positive; otherwise, it will return 0.
    This is a common activation function used in neural networks.
    """
    try:
        return Z > 0
    except Exception as e:
        print("Error in ReLu:", e)
        raise

def softmax(Z):
    """
    Softmax activation function.
    It converts the input values into probabilities that sum to 1.
    This is typically used in the output layer of a classification model.
    ans = exp(Z) / sum(exp(Z))

    Here axis and keepdims are used to ensure there is no overflow and the sum is computed correctly across the specified axis.
    here exp is e^Z
    """
    try:
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)
    except Exception as e:
        print("Error in softmax:", e)
        raise

def forward_prop(W1, W2, b1,b2, X):
    """
    Forward propagation function.
    It computes the output of the neural network given the input X and the parameters W1, W2, b1, b2.
    
    The steps are:
    1. Compute Z1 = W1 * X + b1
    2. Apply ReLU activation to Z1 to get A1
    3. Compute Z2 = W2 * A1 + b2
    4. Apply softmax activation to Z2 to get A2 (the final output)
    """
    try:
        Z1 = np.dot(W1, X) + b1
        A1 = ReLu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = softmax(Z2)

        # Used During Debugging and writing the code
        # print("After Forwar Propogation:-")
        # print("Z1:", Z1)
        # print("A1:", A1)
        # print("Z2:", Z2)
        # print("A2:", A2)

        return Z1, A1, Z2, A2
    except Exception as e:
        print("Error in forward_prop:", e)
        raise

def one_hot(Y):
    """
    One-hot encoding function.
    Converts a vector of class labels into a one-hot encoded 2D array.

    The steps are:
    1. Create a zero matrix of shape (number of samples, number of classes)
    2. For each sample, set the column corresponding to the class label to 1
    """
    try:
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T 

        return one_hot_Y
    except Exception as e:
        print("Error in one_hot:", e)
        raise


def back_prop(Z1,A1,Z2, A2,W2,X,Y, m):
    """
        Read the read.md file or the documentation or the youtube vide for the maths behind this.
        From the output layer to the hidden layer and then to the input layer,
        we calculate the gradients of the loss function with respect to the weights and biases.

            
        Here axis and keepdims are used to ensure there is no overflow and the sum is computed correctly across the specified axis.
    """
    try:
        
        # m= Y.size()

        one_hot_Y = one_hot(Y)
        dZ2 = A2 - one_hot_Y
        
        dW2 = 1/m *np.dot(dZ2, A1.T)
        db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

        
        dZ1 = np.dot(W2.T, dZ2)*(Z1)

        dW1 = 1/m * np.dot(dZ1, X.T)
        db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2
    except Exception as e:
        print("Error in back_prop:", e)
        raise



def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate=0.01):
    """
    Read the read.md file or the documentation or the youtube vide for the maths behind this.
    This function updates the parameters of the model using the gradients computed during backpropagation.
    The parameters are updated using the formula:  
    W = W - learning_rate * dW
    b = b - learning_rate * db     
    """
    
    try:
        W1 =  W1 - learning_rate * dW1
        b1 =  b1 - learning_rate * db1
        W2 =  W2 - learning_rate * dW2 
        b2 =  b2 - learning_rate * db2

        return W1, b1, W2, b2
    except Exception as e:
        print("Error in update_params:", e)
        raise

def get_predictions(A2):
    """
        This function generates predictions from the output of the softmax layer.

        The softmax layer produces a probability distribution across all classes.
        The class with the highest probability is selected as the final prediction.

        np.argmax(A2, 0) returns the index of the maximum probability in each column (sample).
    """
    try:
        return np.argmax(A2, 0)
    except Exception as e:
        print("Error in get_predictions:", e)
        raise


def get_accuracy(predictions, Y):
    """
        This function computes the accuracy of predictions made by the model.

        Accuracy is calculated as:
            (Number of Correct Predictions) / (Total Number of Samples)

        Parameters:
            predictions: A vector of predicted class labels.
            Y: A vector of true class labels.

        Returns:
            A float value representing the accuracy (between 0 and 1).
    """

    try:
        print("Predictions:", predictions)
        print("Y:", Y)

        return np.sum(predictions == Y) / Y.size
    except Exception as e:
        print("Error in get_accuracy:", e)
        raise

def gradient_descent(X,Y, iteration,m, learning_rate=0.01):
    """
        This function performs gradient descent to train the neural network.

        Steps involved:
        1. Initialize weights and biases.
        2. Loop over the number of iterations:
            a. Perform forward propagation to get predictions.
            b. Compute gradients via backpropagation.
            c. Update parameters using the gradients and learning rate.
        3. Optionally print accuracy every 10 iterations to monitor performance.

        Parameters:
            X: Input features (shape: 784 x m).
            Y: True labels.
            iteration: Number of iterations to train the model.
            learning_rate: Learning rate used in parameter updates.
    """
    try:
        W1, b1, W2, b2 = init_params()

        for i in range(iteration):
            print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            Z1, A1, Z2, A2 = forward_prop(W1, W2, b1,b2,X)
            dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X,Y,m)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
            print("Iteration:", i)
            if i % 10 == 0:
                # print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                print("Accuracy:- ", get_accuracy(get_predictions(A2), Y))
                # print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        
        return W1, b1, W2, b2

    except Exception as e:
        print("Error in gradient_descent:", e)
        raise

# Reading the CSV file into a pandas DataFrame
# The 'r' before the string makes it a raw string, which handles the backslashes in the file path
data = pd.read_csv(r'kaggle_dataset\train.csv')

# ---------------------------------------------------------------------------------------------------------------
# In the Dataset:-
# Each row is a training example.
# The first column is the label (0-9) and the rest are pixel values (0-255).
# the pixels are from 0 to 783 and hence the columns are label + 1 to 784 i.e. 785 colums
# ---------------------------------------------------------------------------------------------------------------

print("Data Headers:-") 
print(data.head())

# To convert the DataFrame to a NumPy array for numerical operations
data = np.array(data)

# ---------------------------------------------------------------------------------------------------------------
# m is the number of rowns and n is the number of columns in the data
# ---------------------------------------------------------------------------------------------------------------
m, n = data.shape 

# ---------------------------------------------------------------------------------------------------------------
# To ensure that our model does not overfit we are doing the below steps:
# 1. We will shuffle the data before splitting it into training and development sets.
# 2. We will use 1000 examples for development and the rest for training.  
# ---------------------------------------------------------------------------------------------------------------
np.random.shuffle(data) # Shuffle before splitting into dev and training sets

# ---------------------------------------------------------------------------------------------------------------
# T is used for Transposing the array
# [0:1000] => This selects rows 0 to 999 (total 1000 rows) from the original data DataFrame.
# [0:1000] => This selects rows from 1000 up to (but not including) m.
# ---------------------------------------------------------------------------------------------------------------
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
# X_dev = X_dev / 255.

data_train = data[1000:].T
Y_train = data_train[0]
X_train = data_train[1:n]
# X_train = X_train / 255.
# _,m_train = X_train.shape


Y_train
print("Y_train:", Y_train)


# TRAINING THE MODEL
# Accuracy was 0.111 which was howing the over fitting and hence changing the learning rate to 0.1
# W1,b1,W2,b2 = gradient_descent(X_train, Y_train, 100, m, 0.01)
W1,b1,W2,b2 = gradient_descent(X_train, Y_train, 100, m, 0.1)
print("FINAL VA:UES OF THE WIEGHTS AND BIASES")
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)