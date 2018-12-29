import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn import metrics
from sklearn import linear_model




plt.interactive(False)
pd.set_option('display.max_columns', None)


#imports our data set, removing columns that we don't care about and converting our factors into category
# data types for future analysis
data = pd.read_csv("D:\students-performance-in-exams\StudentsPerformance.csv")
print(data.head(5))


p = sns.countplot(x="math score", data = data, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=90)

#plt.show()


train = pd.get_dummies(data, columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])
trainlabel = train['reading score']
trainlabel = trainlabel[0:800]

test = train[801:-1]
testlabel = test['reading score']

train.drop(['reading score'], axis=1, inplace=True)
test.drop(['reading score'], axis=1, inplace=True)
train = train[:800]

print(train.shape)
print(test.shape)

#clf = linear_model.SGDRegressor( loss='squared_loss', max_iter=5e5, verbose=1, learning_rate= 'optimal', penalty=None)
clf = linear_model.ElasticNet( max_iter=5e8, tol=0.000001, alpha= 0.0001)

clf.fit(train, trainlabel)

results = clf.predict(test)
results = np.transpose(results)

accuracyr2 = metrics.r2_score(testlabel, results)
print('R^2 of the model', accuracyr2)

print(clf.coef_)
print(train.columns.values)

