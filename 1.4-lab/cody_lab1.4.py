import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None) # Read in
# the data from the illinois state website

df.head() # Take a look at it, make sure it's reading properly
# After reading the documentation, it seems like the target field should
# be whether or not a sample tested malignent or benign. The data fields that follow
# this field will be the features. The specifics of what these numbers indicate
# is largely irrelevant for our purposes. All we need to do is use them to make a model
# that will evaluate the same criteria in future patients.
df['mal_or_ben'] = df[1].map({'B':0, 'M':1})
df.mal_or_ben.head()

target = df.mal_or_ben# Set the target field

features = df.ix[:, 2:32] # Set the features
features.head
target.head()



from sklearn import linear_model

X = features
y = target

lr = linear_model.LogisticRegression()
lr.fit(X,y)
lr.score(X,y)

# The above is a completed logisticRegression model for the data set above. We can test it with a random data point from the features
# DataFrame

lr.predict(features.ix[15,]) # This predicts whether the data at row 15 is representative of a malignant tumor sample.
# We can confirm this by looking at row 15 in the original.
df.ix[15,]

# Try the above using KNearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
knn.score(X,y)
