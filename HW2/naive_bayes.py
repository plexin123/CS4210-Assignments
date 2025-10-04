#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []
outlook = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temperature = {'Hot': 3 , 'Cool': 1, 'Mild': 2 }
humidity = {'High':2 ,'Normal':1}
wind = {'Strong': 2, 'Weak': 1 }
clas = {'Yes': 1, 'No': 0}
    
#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
Xt = []
Yt = []
for row in dbTraining:
    Xt.append([
        outlook[row[1]],
        temperature[row[2]],
        humidity[row[3]],
        wind[row[4]],
    ])
    Yt.append([
        clas[row[5]]
    ]
    )

    
#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf.fit(Xt,Yt)  
 #Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())
print("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + 
      "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence")

reverse_clas = {1: 'Yes', 0: 'No'}
for row in dbTest:
    # Transform test instance to numbers
    test_instance = [[
        outlook[row[1]],
        temperature[row[2]],
        humidity[row[3]],
        wind[row[4]]
    ]]
    
    # Get probability predictions: clf.predict_proba returns [[prob_class0, prob_class1]]
    probabilities = clf.predict_proba(test_instance)[0]
    
    # Get the predicted class
    prediction = clf.predict(test_instance)[0]
    predicted_class = reverse_clas[prediction]
    
    # Get confidence (maximum probability)
    confidence = max(probabilities)
    
    # Only print if confidence >= 0.75
    if confidence >= 0.75:
        print(f"{row[0]:<15}{row[1]:<15}{row[2]:<15}{row[3]:<15}{row[4]:<15}"
              f"{predicted_class:<15}{confidence:.4f}")
