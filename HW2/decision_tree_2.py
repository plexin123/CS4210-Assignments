#-------------------------------------------------------------------------
# AUTHOR: Paul Puma
# FILENAME: decision_tree.py
# SPECIFICATION: Train decision trees on contact lens data and evaluate accuracy
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    df_training = pd.read_csv(ds)
    for _, row in df_training.iterrows():
        dbTraining.append(row.tolist())

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    for row in dbTraining:
        # Age: Young=1, Prepresbyopic=2, Presbyopic=3
        age = 1 if row[0] == 'Young' else (2 if row[0] == 'Prepresbyopic' else 3)
        
        # Spectacle: Myope=1, Hypermetrope=2
        spectacle = 1 if row[1] == 'Myope' else 2
        
        # Astigmatism: No=1, Yes=2
        astigmatism = 1 if row[2] == 'No' else 2
        
        # Tear: Normal=1, Reduced=2
        tear = 1 if row[3] == 'Normal' else 2
        
        X.append([age, spectacle, astigmatism, tear])

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    for row in dbTraining:
        Y.append(1 if row[4] == 'Yes' else 2)

    #Loop your training and test tasks 10 times here
    accuracy_sum = 0
    
    for i in range (10):

       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
       clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
       clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest (already done above)
       correct_predictions = 0
       total_predictions = 0

       for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           
           # Age: Young=1, Prepresbyopic=2, Presbyopic=3
           age = 1 if data[0] == 'Young' else (2 if data[0] == 'Prepresbyopic' else 3)
           
           # Spectacle: Myope=1, Hypermetrope=2
           spectacle = 1 if data[1] == 'Myope' else 2
           
           # Astigmatism: No=1, Yes=2
           astigmatism = 1 if data[2] == 'No' else 2
           
           # Tear: Normal=1, Reduced=2
           tear = 1 if data[3] == 'Normal' else 2
           
           class_predicted = clf.predict([[age, spectacle, astigmatism, tear]])[0]
           
           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           true_label = 1 if data[4] == 'Yes' else 2
           
           if class_predicted == true_label:
               correct_predictions += 1
           total_predictions += 1

       #Calculate accuracy for this run
       accuracy = correct_predictions / total_predictions
       accuracy_sum += accuracy

    #Find the average of this model during the 10 runs (training and test set)
    average_accuracy = accuracy_sum / 10

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f"Final accuracy when training on {ds}: {average_accuracy:.1f}")