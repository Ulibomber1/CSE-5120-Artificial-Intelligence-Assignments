# Import libraries
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import pickle
import pandas as pd
import numpy as np

# In this section, you can use a search engine to look for the functions that will help you implement the following steps

# Load dataset and show basic statistics
pulsarData = pd.read_csv('pulsar_stars.csv', sep=', |,', dtype=np.float64, engine="python")
# 1. Show dataset size (dimensions)
print('Data Size: ' + str(len(pulsarData)))
# 2. Show what column names exist for the 9 attributes in the dataset
print('\nAttribute Names:')
print(pulsarData.dtypes)
# 3. Show the distribution of target_class column
# 4. Show the percentage distribution of target_class column
neg_class, pos_class = 0, 0
for entry in pulsarData['target_class']:
    if entry == 0:
        neg_class += 1
    else:
        pos_class += 1
print(f'\nDistribution:\n Negative: {neg_class} ({(neg_class/len(pulsarData))*100}%)\n Positive: {pos_class} ({(pos_class/len(pulsarData))*100}%)\n')

# Separate predictor variables from the target variable (X and y as we did in the class)
X = pulsarData.iloc[:, :-1]
Y = pulsarData.iloc[:,-1]

# Create train and test splits for model development. Use the 80% and 20% split ratio
# Name them as X_train, X_test, y_train, and y_test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)

# Standardize the features (Import StandardScaler here)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Below is the code to convert X_train and X_test into data frames for the next steps
cols = X.columns
X_train = pd.DataFrame(X_train, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd
X_test = pd.DataFrame(X_test, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd

# Train SVM with the following parameters.
   #    1. RBF kernel
   #    2. C=10.0 (Higher value of C means fewer outliers)
   #    3. gamma = 0.3
classifier = SVC(kernel='rbf', C=10.0, gamma=0.3)
classifier.fit(X_train, Y_train)

# Test the above developed SVC on unseen pulsar dataset samples
Y_prediction = classifier.predict(X_test)

# compute and print accuracy score
confusionMatrix = confusion_matrix(Y_test, Y_prediction)
print(f"\nConfusion Matrix:\n {confusionMatrix}")
print(f"\nAccuracy:\n{accuracy_score(Y_test, Y_prediction)}")

# Save your SVC model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment
with open('PulsarClassifier.sav', 'wb') as file:
    pickle.dump(classifier, file)

# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix
#cm = [[]]
# Below are the metrics for computing classification accuracy, precision, recall and specificity
#TP = cm[0,0]
#TN = cm[1,1]
#FP = cm[0,1]
#FN = cm[1,0]



# Compute Precision and use the following line to print it
#precision = 0 # Change this line to implement Precision formula
#print('Precision : {0:0.3f}'.format(precision))


# Compute Recall and use the following line to print it
#recall = 0 # Change this line to implement Recall formula
#print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
#specificity = 0 # Change this line to implement Specificity formula
#print('Specificity : {0:0.3f}'.format(specificity))

