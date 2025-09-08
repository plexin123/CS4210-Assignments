#-------------------------------------------------------------------------
# AUTHOR: Paul Puma
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #1
# TIME SPENT: how long it took you to complete the assignment 
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

##Like a hash table:

Age = {
    'Young': 0,
    'Prepresbyopic': 1,
    'Presbyopic': 2
   }
Spectacle = {
    'Myope': 0,
    'Hypermetrope': 1
}

Astigmatism = {
    'No': 0,
    'Yes': 1
}

Tear = {
   'Reduced' : 0,
   'Normal' : 1
}




#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)



#encode the original categorical training features into numbers and add to the 4D array X.
#--> add your Python code here
# X =
for row in (db):
   #temporal list that contains the values encoded 
   temp = [
      Age[row[0]],
      Spectacle[row[1]],
      Astigmatism[row[2]],
      Tear[row[3]],
   ]
   X.append(temp)

#encode the original categorical training classes into numbers and add to the vector Y.
#--> add your Python code here
# Y =
class_result = {
   'No' : 0,
   'Yes' :1,
}
for rows in db:
  tmp = [
     class_result[row[4]]
  ]
  Y.append(tmp)

#fitting the decision tree to the data using entropy as your impurity measure
#--> addd your Python code here
clf = tree.DecisionTreeClassifier(criterion= "entropy")
clf = clf.fit(X,Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()