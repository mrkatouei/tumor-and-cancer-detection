import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


cancer = load_breast_cancer()
dataFrame = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

#Graphs
sns.pairplot(dataFrame, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])

#Heat Map
plt.figure(figsize = (20, 20))
sns.heatmap(dataFrame.corr(), annot = True)

#Count Plot
sns.countplot(dataFrame['target'])

#Scatter Plot
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = dataFrame)

x = dataFrame.drop(['target'], axis = 1)
y = dataFrame['target']
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 42)

svcModel = SVC()

minTrain = xTrain.min()
rangeTrain = (xTrain - minTrain).max()
xTrainScaled = (xTrain - minTrain) / rangeTrain

minTest = xTest.min()
rangeTest = (xTest - minTest).max()
xTestScaled = (xTest - minTest) / rangeTest 

svcModel.fit(xTrainScaled, yTrain)
yPredictScaled = svcModel.predict(xTestScaled)

Grid = GridSearchCV(SVC(), paramGrid, refit = True, verbose = 2)
Grid.fit(xTrainScaled, yTrain)
print(Grid.best_params_)
gridPredictions = Grid.predict(xTestScaled)

cn = confusion_matrix(yTest, gridPredictions)
sns.heatmap(cn, annot = True)

print(classification_report(yTest, gridPredictions))
