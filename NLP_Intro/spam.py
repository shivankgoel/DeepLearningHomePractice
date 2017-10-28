from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

data = pd.read_csv('data/spam/spambase.data').as_matrix()
np.random.shuffle(data)

x = data[:,:-1]
y = data[:,-1]


xtrain = x[:-100,:]
xtest = x[-100:,:]
ytrain = y[:-100]
ytest = y[-100:]

model = MultinomialNB()
model.fit(xtrain,ytrain)
model.score(xtest,ytest)


from sklearn.ensemble import AdaBoostClassifier

model2 = AdaBoostClassifier()
model2.fit(xtrain,ytrain)
model2.score(xtest,ytest)