import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.model_selection import cross_val_score


df = pd.read_csv('/home/shehab/Downloads/Recruiting_Task_InputData.csv')
test_df = []
test_df = df[['age','earnings']]
life_sty = pd.get_dummies(df.lifestyle)
fam_stat = pd.get_dummies(df['family status'])
car_     = pd.get_dummies(df.car)
sports_  = pd.get_dummies(df.sports)
liv_area = pd.get_dummies(df['living area'])

test_df = pd.concat([test_df, life_sty, fam_stat, car_, sports_, liv_area], axis = 1)
test_df['label'] = df['label'].apply(lambda x: 1 if (x == 'response')  else 0)
x = test_df.iloc[:,:-1].values
y = test_df['label'].values
X = x

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.15, random_state=4)
#decision tree object (model)
decTree   = DecisionTreeClassifier(criterion="entropy", max_depth = 2,random_state=4).fit(X_train,y_train)
predTree  = decTree.predict(X_test)
print(metrics.accuracy_score(y_test, predTree))


from  io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
dot_data = StringIO()
filename = "predtree.png"
featureNames = test_df.columns[0:14]
out = tree.export_graphviz(decTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(df.label), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
