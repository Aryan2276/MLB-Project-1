import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)

data = pd.read_csv("Iris.csv")
data.head()
print(data)


print(data["Species"].value_counts())

sns.FacetGrid(data, hue="Species", height=6).map(sns.scatterplot, "PetalLengthCm", "SepalWidthCm").add_legend()
plt.show()

#Converting categorical variables into numbers  
flower_mapping = {'Iris-setosa': 0,'Iris-versicolor': 1,'Iris-virginica': 2}
data["Species"] = data["Species"].map(flower_mapping)
data.head()

#Preparing inputs and outputs
X=data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
y=data['Species'].values

#logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X, y)

#Accuracy
score = model.score(X,y)
print(score)


#make predictions
expected = y
predicted = model.predict(X)
predicted

#summarize the fit of the model
from sklearn import metrics
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))