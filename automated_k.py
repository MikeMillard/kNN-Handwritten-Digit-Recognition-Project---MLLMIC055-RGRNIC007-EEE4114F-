# Imports
import os

print("\nRunning 4-fold cross validation tests for k = [3, 5, 7, 9, 11, 13, 15]...")
# For loop to run script with different k values
for i in range(3, 16, 2):
    csv_name = "BigTrain.csv"
    train_mode = False
    command = "python kNN_project.py " + str(i) + " " + csv_name + " " + train_mode
    os.system(command)