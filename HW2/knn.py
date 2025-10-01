#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

wrongpred = 0

#Loop your data to allow each instance to be your test set
for i in range(len(db)):
    
    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    Y = []
    for j in range(len(db)):
        if i == j:
            continue
        current_row = db[j]
        features = []
        for val in current_row[:-1]:
            features.append(float(val))
        X.append(features)
        clv = current_row[-1]
        if clv == "ham":
            Y.append(float(1))
        else:
            Y.append(float(0))
    test_row = db[i]
    Xc = []
    for value in test_row[:-1]:
        Xc.append(value)
    cls = test_row[-1]
    if cls == "ham":
        actual_label = (float(1))
    else:
        actual_label = (float(0))
    
    ctrain = KNeighborsClassifier(n_neighbors=1, metric= 'euclidean')
    ctrain.fit(X,Y)
    
    prelabel = ctrain.predict([Xc])[0]
    
    if prelabel != actual_label:
        wrongpred = wrongpred + 1

    total_tests = len(db)
    error_rate = float(wrongpred)/ float(total_tests)

print("Error Rate: " + str(error_rate))
print("Wrong predictions: " + str(wrongpred) + " out of " + str(total_tests))
print("Accuracy: " + str(1.0 - error_rate))


