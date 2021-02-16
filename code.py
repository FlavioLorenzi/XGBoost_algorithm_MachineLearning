# original work at https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
# Flavio Lorenzi

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
dataset = loadtxt('pimaDataset.csv', delimiter=",")
#print(dataset) #to see the dataset in the terminal

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


##################
#### TRAINING ####
##################

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)


print(model)	# to see params of the model


##################
#### TESTING #####
##################

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)

#see the accuracy
print("Accuracy: %.2f%%" % (accuracy * 100.0))








