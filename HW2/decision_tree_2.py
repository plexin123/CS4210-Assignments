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

dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    df_training = pd.read_csv(ds)
    for _, row in df_training.iterrows():
        dbTraining.append(row.tolist())

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    for row in dbTraining:

        age = 1 if row[0] == 'Young' else (2 if row[0] == 'Prepresbyopic' else 3)
        

        spectacle = 1 if row[1] == 'Myope' else 2
        

        astigmatism = 1 if row[2] == 'No' else 2
        
 
        tear = 1 if row[3] == 'Normal' else 2
        
        X.append([age, spectacle, astigmatism, tear])

 
    for row in dbTraining:
        Y.append(1 if row[4] == 'Yes' else 2)
 
    accuracy_sum = 0
    
    for i in range (10):

 
       clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
       clf = clf.fit(X, Y)

       correct_predictions = 0
       total_predictions = 0

       for data in dbTest:

           age = 1 if data[0] == 'Young' else (2 if data[0] == 'Prepresbyopic' else 3)
           

           spectacle = 1 if data[1] == 'Myope' else 2

           astigmatism = 1 if data[2] == 'No' else 2
           

           tear = 1 if data[3] == 'Normal' else 2
           
           class_predicted = clf.predict([[age, spectacle, astigmatism, tear]])[0]
           
 
           true_label = 1 if data[4] == 'Yes' else 2
           
           if class_predicted == true_label:
               correct_predictions += 1
           total_predictions += 1

                        
       accuracy = correct_predictions / total_predictions
       accuracy_sum += accuracy

    average_accuracy = accuracy_sum / 10
    print(f"Final accuracy when training on {ds}: {average_accuracy:.1f}")