import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

import sklearn
from sklearn import tree

"""
# Decision Tree
CART (Classification  and Regression Tree)

## What is Decision Tree?
***
* Supervised Learning
* Works for BOTH Classification and Regression
* Foundation of Random Forests
* Attractive because of interpretability

***

Decision Tree works by:
* Split based on set impurity criteria
* Stopping criteria

***

Source: [Scikit-Learn](http://scikit-learn.org/stable/modules/tree.html#tree)

Some **advantages** of decision trees are:
* Simple to understand and to interpret. Trees can be visualised.
* Requires little data preparation. 
* Able to handle both numerical and categorical data.
* Possible to validate a model using statistical tests. 
* Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.

The **disadvantages** of decision trees include:
* Overfitting. Mechanisms such as pruning (not currently supported), 
  setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are 
  necessary to avoid this problem.
* Decision trees can be unstable. Mitigant: Use decision trees within an ensemble.
* Cannot guarantee to return the globally optimal decision tree. Mitigant: Training multiple trees in an ensemble learner
* Decision tree learners create biased trees if some classes dominate. Recommendation: Balance the dataset prior to fitting

***

## Questions:

1. What is a decision tree?
2. Where can you apply decision tree to? numerical problems or categorical problems?
3. Decision tree is also know by what other name?
4. How does a decision tree work?
5. Decision Tree is a foundation of what machine learning algorithm
6. List and explain 3 advantages of decision tree
7. List and explain 3 disadvantages of decision tree

# Classification
## Training a Decision Tree with Scikit-Learn Library
"""


X = [[0, 0], [1, 2]]
y = [0, 1]

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, y)

clf.predict([[2., 2.]])
# Predicted to be 1

clf.predict_proba([[2., 2.]])
# prob of 0 - 0; prob to be 1 - 1

clf.predict([[0.4, 1.2]])
# Predicted to be 0

clf.predict_proba([[0.4, 1.2]])
# prob of 0 - 1.0; prob to be 1 - 0

clf.predict_proba([[0, 0.2]])

"""
`DecisionTreeClassifier` is capable of both binary (where the labels are [-1, 1]) classification and multiclass (where the labels are [0, …, K-1]) classification.

## Applying to Iris Dataset
"""

from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

print(iris.data[0:5])

print(iris.feature_names)

X = iris.data[:, 2:]

y = iris.target

y

clf = tree.DecisionTreeClassifier(random_state=42)

clf = clf.fit(X, y)

"""
### Export_graphviz
Need to install graphviz first
pip3 install graphviz
brew install graphviz
"""

from sklearn.tree import export_graphviz

export_graphviz(clf,
                out_file="/Users/yuawong/Documents/GitHub/udemy_repo/The_Complete_Machine_Learning_Course_with_Python/"
                         "6_Tree/tree.dot",
                feature_names=iris.feature_names[2:],
                class_names=iris.target_names,
                rounded=True,
                filled=True)

"""Run the following line on your command prompt

## Graphviz
"""

import graphviz

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names[2:],
                                class_names=iris.target_names,
                                rounded=True,
                                filled=True)

graph = graphviz.Source(dot_data)

# To visualise
graph.view()
"""graph
## Visualise the Decision Boundary
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt

df = sns.load_dataset('iris')
df.head()

col = ['petal_length', 'petal_width']
X = df.loc[:, col]

species_to_num = {'setosa': 0,
                  'versicolor': 1,
                  'virginica': 2}
df['tmp'] = df['species'].map(species_to_num)
y = df['tmp']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

Xv = X.values.reshape(-1, 1)
h = 0.02
x_min, x_max = Xv.min(), Xv.max() + 1
y_min, y_max = y.min(), y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

fig = plt.figure(figsize=(8, 8))
ax = plt.contourf(xx, yy, z, cmap='afmhot', alpha=0.3)
plt.scatter(X.values[:, 0], X.values[:, 1], c=y, s=80, 
            alpha=0.9, edgecolors='g')

"""
# Decision Tree Learning

* [ID3](https://en.wikipedia.org/wiki/ID3_algorithm) (Iterative Dichotomiser 3)
* [C4.5](https://en.wikipedia.org/wiki/C4.5_algorithm) (successor of ID3)
* CART (Classification And Regression Tree)
* [CHAID](http://www.statisticssolutions.com/non-parametric-analysis-chaid/) (Chi-squared Automatic Interaction Detector). by [Gordon Kass](https://en.wikipedia.org/wiki/Chi-square_automatic_interaction_detection).

## Tree algorithms: ID3, C4.5, C5.0 and CART


* ID3 (Iterative Dichotomiser 3) was developed in 1986 by Ross Quinlan. The algorithm creates a multiway tree, finding for each node (i.e. in a greedy manner) the categorical feature that will yield the largest information gain for categorical targets. Trees are grown to their maximum size and then a pruning step is usually applied to improve the ability of the tree to generalise to unseen data.


* C4.5 is the successor to ID3 and removed the restriction that features must be categorical by dynamically defining a discrete attribute (based on numerical variables) that partitions the continuous attribute value into a discrete set of intervals. C4.5 converts the trained trees (i.e. the output of the ID3 algorithm) into sets of if-then rules. These accuracy of each rule is then evaluated to determine the order in which they should be applied. Pruning is done by removing a rule’s precondition if the accuracy of the rule improves without it.


* C5.0 is Quinlan’s latest version release under a proprietary license. It uses less memory and builds smaller rulesets than C4.5 while being more accurate.


* CART (Classification and Regression Trees) is very similar to C4.5, but it differs in that it supports numerical target variables (regression) and does not compute rule sets. CART constructs binary trees using the feature and threshold that yield the largest information gain at each node.


* CHAID (Chi-squared Automatic Interaction Detector). by Gordon Kass. Performs multi-level splits when computing classification trees. Non-parametric. Does not require the data to be normally distributed. 

scikit-learn uses an optimised version of the CART algorithm.
"""


# Gini Impurity
"""
[Gini Impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)

A measure of purity / variability of categorical data

As a side note on the difference between [Gini Impurity and Gini Coefficient](https://datascience.stackexchange.com/questions/1095/gini-coefficient-vs-gini-impurity-decision-trees)

* No, despite their names they are not equivalent or even that similar.
* **Gini impurity** is a measure of misclassification, which applies in a multiclass classifier context.
* **Gini coefficient** applies to binary classification and requires a classifier that can in some way rank examples according to the likelihood of being in a positive class.
* Both could be applied in some cases, but they are different measures for different things. Impurity is what is commonly used in decision trees.


Developed by [Corrado Gini](https://en.wikipedia.org/wiki/Corrado_Gini) in 1912

Key Points:
* A pure node (homogeneous contents or samples with the same class) will have a Gini coefficient of zero
* As the variation increases (heterogeneneous classes or increase diversity), Gini coefficient increases and approaches 1.

$$Gini=1-\sum^r_j p^2_j$$

$p$ is the probability (often based on the frequency table)

<img src='img//tree_gini_imp.png' width=50%>

# Entropy

[Wikipedia](https://en.wikipedia.org/wiki/Entropy_(information_theory)

The entropy can explicitly be written as

$${\displaystyle \mathrm {H} (X)=\sum _{i=1}^{n}{\mathrm {P} (x_{i})\,\mathrm {I} (x_{i})}=-\sum _{i=1}^{n}{\mathrm {P} (x_{i})\log _{b}\mathrm {P} (x_{i})},}$$

where `b` is the base of the logarithm used. Common values of `b` are 2, Euler's number `e`, and 10

# Which should I use?

[Sebastian Raschka](https://sebastianraschka.com/faq/docs/decision-tree-binary.html)

* They tend to generate similar tree
* Gini tends to be faster to compute
"""

def gini(p):
    return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))
def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))
def error(p):
    return 1 - np.max([p, 1 - p])


# From 0 - 1, every 0.01, will have 100 num
x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]

sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]


fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                          ['Entropy', 'Entropy (scaled)',
                           'Gini Impurity',
                           'Misclassification Error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray',
                           'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab,
                   linestyle=ls, lw=2, color=c)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()

"""# Information Gain

* Expected reduction in entropy caused by splitting 

* Keep splitting until you obtain a as close to homogeneous class as possible
"""

# Regression
from sklearn import tree

X = [[0, 0], [3, 3]]
y = [0.75, 3]

tree_reg = tree.DecisionTreeRegressor(random_state=42)

tree_reg = tree_reg.fit(X, y)

# Predicted to be 0.75
tree_reg.predict([[1.5, 1.5]])


# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

# depth = 5 leads to over-fitting

"""
dot_data = tree.export_graphviz(regr_1, out_file=None,
                                filled=True)
graph = graphviz.Source(dot_data)
graph.view()

dot_data = tree.export_graphviz(regr_2, out_file=None,
                                filled=True)
graph = graphviz.Source(dot_data)
graph.view()
"""

## Regularization
# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure(figsize=(10,8))
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

"""
dot_data = tree.export_graphviz(regr_2, out_file=None,
                                filled=True)
graph = graphviz.Source(dot_data)
graph.view()
"""


# Overfitting
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

X = iris.data[:, 0:2]
y = iris.target
clf = tree.DecisionTreeClassifier(random_state=42)
clf = clf.fit(X, y)

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names[2:],
                                class_names=iris.target_names,
                                rounded=True,
                                filled=True)

"""
graph = graphviz.Source(dot_data)
graph.view()
"""


# Modelling End-to-End with Decision Tree
from sklearn.datasets import make_moons

X_data, y_data = make_moons(n_samples=1000, noise=0.5, random_state=42)

cl1 = tree.DecisionTreeClassifier(random_state=42)
cl2 = tree.DecisionTreeClassifier(min_samples_leaf=10, random_state=42)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

from sklearn.model_selection import GridSearchCV

# params = {'max_leaf_nodes': list(range(2, 50)),
#          'min_samples_split': [2, 3, 4],
#          'min_samples_leaf': list(range(5, 20))}

params = {'min_samples_leaf': list(range(5, 20))}

grid_search_cv = GridSearchCV(tree.DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1)

grid_search_cv.fit(X_train, y_train)

print(grid_search_cv.best_estimator_)

from sklearn.metrics import accuracy_score

y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.83 with gridsearch

cl1.fit(X_train, y_train)
y_pred = cl1.predict(X_test)
accuracy_score(y_test, y_pred)
# Only 0.74
cl1.get_params()

"""
# Where to From Here
## Tips on practical use
* Decision trees tend to overfit on data with a large number of features. Check ratio of samples to number of features
* Consider performing dimensionality reduction (PCA, ICA, or Feature selection) beforehand
* Visualise your tree as you are training by using the export function. Use max_depth=3 as an initial tree depth.
* Use max_depth to control the size of the tree to prevent overfitting.
* Tune `min_samples_split` or `min_samples_leaf` to control the number of samples at a leaf node. 
* Balance your dataset before training to prevent the tree from being biased toward the classes that are dominant. 
  * By sampling an equal number of samples from each class  
  * By normalizing the sum of the sample weights (sample_weight) for each class to the same value.

# References:
1. [Wikipedia - Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
2. [Decision Tree - Classification](http://www.saedsayad.com/decision_tree.htm)
3. [Data Aspirant](http://dataaspirant.com/2017/01/30/how-decision-tree-algorithm-works/)
4. [Scikit-learn](http://scikit-learn.org/stable/modules/tree.html)
5. https://en.wikipedia.org/wiki/Predictive_analytics
6. L. Breiman, J. Friedman, R. Olshen, and C. Stone. Classification and Regression Trees. Wadsworth, Belmont, CA, 1984.
7. J.R. Quinlan. C4. 5: programs for machine learning. Morgan Kaufmann, 1993.
8. T. Hastie, R. Tibshirani and J. Friedman. Elements of Statistical Learning, Springer, 2009.
"""

