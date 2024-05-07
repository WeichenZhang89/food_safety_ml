import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# generate data
# X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=2.5)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');
# plt.show()

# import Obesity Classification.csv
data = pd.read_csv('/Users/small_yellow/Documents/ECS 170/Final Project/mushrooms.csv')
# transfer catergorical date to numerical data
data['cap-shape'].replace(['b', 'c', 'x', 'f', 'k','s'],[0, 1, 2, 3, 4, 5], inplace=True)
data['cap-surface'].replace(['f', 'g', 'y', 's'],[0, 1, 2, 3], inplace=True)
data['class'].replace(['p', 'e'],[0, 1], inplace=True)


# define X and y
X = data[['cap-shape', 'cap-surface']]
y = data['class'] 
# show the scattered plot
plt.scatter(X['cap-shape'], X['cap-surface'],c=y, s=50, cmap='RdBu');
plt.show()


#build the model
model = GaussianNB()
model.fit(X,y)

rng = np.random.RandomState(0)
Xnew = [10, 5] + [150, 31] * rng.rand(8124, 2)
##print("shape",Xnew.shape)
##print(Xnew[0:5,:])


ynew = model.predict(X)
plt.scatter(X['cap-shape'], X['cap-surface'],c=y, s=50, cmap='RdBu');
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.3)
plt.axis(lim)
plt.show()
yprob = model.predict_proba(Xnew ) # predicting probabilities for each label

print(yprob[0:10].round(2))

# show accuracy 
print(metrics.classification_report(ynew, y))

# print the confusion matrix
mat = confusion_matrix(y, ynew)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');

plt.show()