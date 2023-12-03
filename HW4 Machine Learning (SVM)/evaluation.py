# 1. import any required library to load dataset, open files (os), print confusion matrix and accuracy score
import pickle
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# 2. Create test set if you like to do the 80:20 split programmatically or if you have not already split the data at this point
pulsarData = pd.read_csv('pulsar_stars.csv', sep=', |,', dtype=np.float64, engine="python")
X = pulsarData.iloc[:, :-1]
Y = pulsarData.iloc[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)

sc = StandardScaler()
X_test = sc.fit_transform(X_test)

cols = X.columns
X_test = pd.DataFrame(X_test, columns=[cols])
# 3. Load your saved model for pulsar classifier that you saved in pulsar_classification.py via Pikcle
with open('PulsarClassifier.sav', 'rb') as file:
    classifier = pickle.load(file)
# 4. Make predictions on test_set created from step 2
Y_prediction = classifier.predict(X_test)

# 5. use predictions and test_set (X_test) classifications to print the following:
#    1. confution matrix, 2. accuracy score, 3. precision, 4. recall, 5. specificity
#    You can easily find the formulae for Precision, Recall, and Specificity online.
# Get and print confusion matrix
cm = confusion_matrix(Y_test, Y_prediction)
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

accuracy = (TP + TN)/(TP + TN + FP + FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
specificity = TN/(FP+TN)

print(f"\nConfusion Matrix:\n {cm}")
print(f"\nAccuracy:\n{accuracy}")
print(f"\nPrecision:\n{precision}")
print(f"\nRecall:\n{recall}")
print(f"\nSpecificity:\n{specificity}")
