################################################################################################################################
# EEE4114F - DSP/ML Project (Handwritten Digit Image Classification k-Nearest-Neighbours Project)
# Authors: Michael Millard and Nicholas Rogers
# Student Numbers: MLLMIC055 and RGRNIC007
# Date: 18/05/2022
################################################################################################################################

# Format for running this python script from terminal:
# > python kNN_project.py k_value file_name.csv train_mode
# Command has three parameters accessed from sys.argv: k_value (number of neighbours), file_name.csv (containing dataset), 
# and train_mode (either a 0 or a 1 to tell the script whether it is doing cross validation or generalization)

# Imports
import sys
import csv
import numpy as np
from scipy.stats import mode
from sklearn.model_selection import train_test_split

# Task: Correctly classify given images containing hand-written digits into one of 10 classes: 0, 1, 2, ..., 9.
# The code below reads in the csv file containing the dataset from which training, validation and testing sets are chosen.
# MNIST data: Each example has 784 features (pixel values) from a 28x28 image reshaped into a 1D matrix (28 x 28 = 784).
# One row in the csv file contains the data for one example-label pair. The label is given in the first column of the
# row and the remaining 784 columns contain the pixel data that comprises the features of that particular example.

fileName = sys.argv[2]
with open(fileName) as csvFile:
    reader_obj = csv.reader(csvFile)
    count = 0   # Needed to ignore the first row containing the headings of each column
    X = []      # Array for all of the examples in the csv file
    Y = []      # Array for the lables corresponding to each example
    for line in reader_obj:
        features = []
        if (count > 0):
            label = int(line[0])    # Label is the first column of each row
            for element in line[1:]: # The rest of the rows contain the features for that element
                features.append(int(element))
            X.append(features)
            Y.append(label)
            count += 1
        else:   # If reading the first row (headings), increment count and continue
            count += 1
            continue

# 80% of dataset used as training data (further split into 4 subsets for 4-fold cross validation)
# 20% of dataset used as testing data (to assess generalization performance of optimal k value found using cross validation)
# 1st set of tests done on 1200 examples. 2nd set of tests done on 2400 examples.
# k values checked are [3, 5, 7, 9, 11, 13, 15] (done by automated python script)
# 4-fold Cross validation: Splitting training data in subsets of 4

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8) # Split the dataset from the csv file 80/20 (training/testing)
D12, D34, L12, L34 = train_test_split(X_train, Y_train, train_size = 0.5) # Split the training data in two (1st step in 4-fold cross validation)
D1, D2, L1, L2 = train_test_split(D12, L12, train_size = 0.5) # Split the first half of the split above in two
D3, D4, L3, L4 = train_test_split(D34, L34, train_size = 0.5) # Split the second half of the split above in two
# We now have 4 subsets of training data (D1:L1, D2:L2, D3:L3, D4:L4) where DX contains examples and YX contains labels corresponding to those examples

# Add the training example subsets to a dictionary
example_dict = {
    'D1' : D1,
    'D2' : D2,
    'D3' : D3,
    'D4' : D4
}
example_keys = list(example_dict.keys()) # A list of the example dictionary keys

# Add their corresponding labels to a dictionary
label_dict = {
    'L1' : L1,
    'L2' : L2,
    'L3' : L3,
    'L4' : L4
}
label_keys = list(label_dict.keys()) # A list of the label dictionary keys

# All function definitions required to implement a kNN algorithm are contained below
def euclideanDistance(x_a, x_b):
    """
    Calculates the Euclidean distance between two vectors
    
    Arguments:
        x_a (array): shape [n_features, ] a single vector a
        x_b (array): shape [n_features, ] a single vector b
    
    Returns:
        distance (float): distance between vectors x_a and x_b
    """
    
    distance = 0.0 # Float distance to be returned
    difference_vector = np.zeros(len(x_a)); # Creating a vector to hold the difference between features of the same index from different arrays
   
    for i in range(len(x_a)):
        difference_vector[i] = x_a[i] - x_b[i] # Populate difference vector with differences between corresponding features
        
    distance = np.sqrt(np.sum(np.power(difference_vector, 2))) # Distance is the square root of the sum of each difference vector element squared

    return distance

def calculateDistances(x_test, X_in):
    """
    Calculates the distance between a single test example, x_test,
    and a list of examples X_in. 
    
    Args:
        x_test (array): shape [n_features,] a single test example
        X_in (array): shape [n_samples, n_features] a list of examples to compare against.
    
    Returns:
        distance_list (list of float): The list containing the distances       
    """
    
    distance_list = []
    
    for i in range(len(X_in)):
        distance_list.append(euclideanDistance(x_test, X_in[i])) # Use euclidean distance function above to populate list
        
    return distance_list

def kNearestIndices(distance_list, k):
    """
    Determines the indices of the k nearest neighbours
    
    Arguments:
        distance_list (list of float): list of distances between a test point 
            and every training example
        k (int): the number of nearest neighbours to consider
    
    Returns:
        k_nearest_indices (array of int): shape [k,] array of the indices 
            corresponding to the k nearest neighbours
    """
    
    sorted_dist_list = np.array(distance_list) # Turn distance_list into a Numpy array
    k_nearest_indices = np.argsort(sorted_dist_list)[:k] # Trim the arg sorted array to the first k elements

    return k_nearest_indices

def kNearestNeighbours(k_nearest_indices, X_in, Y_in):
    """
    Creates the dataset of k nearest neighbours
    
    Arguments:
        k_nearest_indices (array of int): shape [k,] array of the indices 
            corresponding to the k nearest neighbours
        X_in (array): shape [n_examples, n_features] the example data matrix to sample from
        Y_in (array): shape [n_examples, ] the label data matrix to sample from
    
    Returns:
        X_k (array): shape [k, n_features] the k nearest examples
        Y_k (array): shape [k, ] the labels corresponding to the k nearest examples
    """
    
    X_k = []
    Y_k = []
    
    for i in range(len(k_nearest_indices)):
        X_k.append(X_in[k_nearest_indices[i]]) # Append the kNN examples to an example list
        Y_k.append(Y_in[k_nearest_indices[i]]) # Append the corresponding labels to those kNN examples to a label list

    return X_k, Y_k

def predict(x_test, X_in, Y_in, k):
    """
    Predicts the class of a single test example
    
    Arguments:
        x_test (array): shape [n_features, ] the test example to classify
        X_in (array): shape [n_input_examples, n_features] the example data matrix to sample from
        Y_in (array): shape [n_input_labels, ] the label data matrix to sample from
    
    Returns:
        prediction (array): shape [1,] the number corresponding to the class 
    """
    
    prediction = []
    
    distance_list = calculateDistances(x_test, X_in) # Obtain the distance list
    k_nearest_indices = kNearestIndices(distance_list, k) # Get indices of k nearest elements in distance list
    Y_k = kNearestNeighbours(k_nearest_indices, X_in, Y_in)[1] # Use indices to get kNN examples and their corresponding labels
    prediction = mode(Y_k).mode # Set prediction equal to the mode of the labels (most common class)
    
    return prediction

def predictBatch(X_t, X_in, Y_in, k):
    """
    Performs predictions over a batch of test examples
    
    Arguments:
        X_t (array): shape [n_test_examples, n_features]
        X_in (array): shape [n_input_examples, n_features]
        Y_in (array): shape [n_input_labels, ]
        k (int): number of nearest neighbours to consider
    
    Returns:
        predictions (array): shape [n_test_examples,] the array of predictions
        
    """
    
    predictions = []
    
    for i in range(len(X_t)):
        predictions = np.concatenate((predictions, predict(X_t[i], X_in, Y_in, k)), axis = 0)
        
    return predictions

def accuracy(Y_pred, Y_test):
    """
    Calculates the accuracy of the model 
    
    Arguments:
        Y_pred (array): shape [n_test_examples,] an array of model predictions
        Y_test (array): shape [n_test_labels,] an array of test labels to 
            evaluate the predictions against
    
    Returns:
        accuracy (float): the accuracy of the model
    """
    
    # Initialize required variables
    accuracy = 0.0
    correct_count = 0
    total_count = len(Y_pred)
    
    for i in range(len(Y_pred)):
        if (Y_pred[i] == Y_test[i]):
            correct_count += 1 # Increment correct count if prediction was correct
    
    accuracy = correct_count/total_count # Determine accuracy
    
    return accuracy

def run(X_train, X_test, Y_train, Y_test, k):
    """
    Evaluates the model on the test data
    
    Arguments:
        X_train (array): shape [n_train_examples, n_features]
        X_test (array): shape [n_test_examples, n_features]
        Y_train (array): shape [n_train_examples, ]
        Y_test (array): shape [n_test_examples, ]
        k (int): number of nearest neighbours to consider
    
    Returns:
        test_accuracy (float): the final accuracy of your model 
    """
    
    test_accuracy = 0.0
    predictions = predictBatch(X_test, X_train, Y_train, k)
    test_accuracy = accuracy(predictions, Y_test)
    
    return test_accuracy

# Obtaining the k value and training mode given in the terminal command
k_grade = int(sys.argv[1])
train_mode = bool(int(sys.argv[3])) # Converting the 0/1 to a boolean

if (train_mode): # If in training mode (cross-validation)
    print("\nImplementing 4-fold cross validation for k = {}:".format(k_grade))
    accuracies = [] # Create an array of accuracies for each of the 4 combinations of the training/validation datasets
    for i in range(4):
        # Splitting the training data into 3 training sets and 1 validation set
        valid_example_key = example_keys[i] # The validation example key for this iteration
        temp_keys = example_keys.copy() # Create a copy of the example keys to remove from later
        temp_keys.remove(valid_example_key) # Remove the validation key, leaving the other 3 training example set keys
        validation_set = example_dict.get(valid_example_key) # Choose validation example set corresponding to the validation example key
        temp_dict = example_dict.copy()     # Create a copy of the example dictionary to remove from later
        temp_dict.pop(valid_example_key)    # Remove the validation example set, leaving the other 3 training example sets
        training_set = []
        for item in temp_dict.values(): # Put all of the training examples in one array
            training_set += item                

        # Splitting the training data into 3 training sets and 1 validation set
        valid_label_key = label_keys[i] # The validation label key for this iteration
        validation_labels = label_dict.get(valid_label_key) # Choose validation label set corresponding to the validation label key
        temp_dict = label_dict.copy()   # Create a copy of the label dictionary to remove from later
        temp_dict.pop(valid_label_key)  # Remove the validation label set, leaving the other 3 training label sets
        training_labels = []
        for item in temp_dict.values(): # Put all of the training labels in one array
            training_labels += item
        
        # Now that the examples and labels are in arrays (for both training and validation sets) insert them into the kNN functions
        test_accuracy = run(training_set, validation_set, training_labels, validation_labels, k_grade)
        predictions_grade = predictBatch(validation_set, training_set, training_labels, k_grade)
        test_accuracy_grade = accuracy(predictions_grade, validation_labels)
        accuracies.append(round(test_accuracy_grade*100, 2))

        # Formatting the print statement for ease of following and understanding the output
        print("k value: {0}, training datasets: {1}, {2}, {3}, validation dataset: {4}, accuracy: {5}%".format(k_grade, temp_keys[0], temp_keys[1], temp_keys[2], valid_example_key, round(test_accuracy_grade*100, 2)))

    # Compute the average accuracy from the 4-fold cross validation for the k value used
    average_acc = sum(accuracies)/len(accuracies)
    print("For k = {0} the average accuracy was {1}%".format(k_grade, round(average_acc, 2)))

else: # Otherwise run the optimal k value on the test set to see how well it generalizes   
    print("Testing generalization Performance for optimal k value (k = {})...".format(k_grade))
    print("Example split: 80/20 to training and testing, respectively\n")

    # Input the training and testing datasets into the kNN functions along with the optimal k value
    test_accuracy = run(X_train, X_test, Y_train, Y_test, k_grade)
    predictions_grade = predictBatch(X_test, X_train, Y_train, k_grade)
    test_accuracy_grade = accuracy(predictions_grade, Y_test)
    print("Generalization performance: k value: {0}, accuracy: {1}%".format(k_grade, round(test_accuracy_grade*100, 2)))