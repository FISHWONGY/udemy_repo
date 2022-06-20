import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

"""
# Principal Component Analysis (PCA) - Linear
[scikit-learn Doc](http://scikit-learn.org/stable/modules/decomposition.html#pca)
[scikit-learn Parameters](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)
* Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.


1901 by Karl Pearson
* Unsupervised Machine Learning
[Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)


* Statistical procedure that utilise [orthogonal transformation](https://en.wikipedia.org/wiki/Orthogonal_transformation) technology
* Convert possible correlated features (predictors) into linearly uncorrelated features (predictors) called **principal components**

* \# of principal components <= number of features (predictors)

* First principal component explains the largest possible variance
* Each subsequent component has the highest variance subject to the restriction that it must be orthogonal to the preceding components. 
* A collection of the components are called vectors.
* Sensitive to scaling

**Note:**
* Used in exploratory data analysis (EDA) 
* Visualize genetic distance and relatedness between populations. 

* Method
  * Eigenvalue decomposition of a data covariance (or correlation) matrix
  * Singular value decomposition of a data matrix (After mean centering / normalizing ) the data matrix for each attribute.

* Output
  * Component scores, sometimes called **factor scores** (the transformed variable values)
  * **loadings** (the weight)

* Data compression and information preservation 
* Visualization
* Noise filtering
* Feature extraction and engineering
"""

rnd_num = np.random.RandomState(42)
X = np.dot(rnd_num.rand(2, 2), rnd_num.randn(2, 500)).T

X

X[:, 0] = - X[:, 0]

plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal')

"""## Principal Components Identification"""

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)

print(pca.components_)

print(pca.explained_variance_)

print(pca.explained_variance_ratio_)
# The first factor explained 92.7% of the data, the 2nd factor explained 72%


plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
# plot data
for k, v in zip(pca.explained_variance_, pca.components_):
    vec = v * 3 * np.sqrt(k)
    
    ax = plt.gca()
    arrowprops=dict(arrowstyle='<-',
                    linewidth=4,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', pca.mean_, pca.mean_ + vec, arrowprops=arrowprops)
    ax.text(-0.90, 1.2, 'PC1', ha='center', va='center', rotation=-42, size=12)
    ax.text(-0.1, -0.6, 'PC2', ha='center', va='center', rotation=50, size=12)
plt.axis('equal')


"""* Two principal components
* Length denotes the significance 
This transformation from data axes to principal axes is as an affine transformation, which basically means it is 
composed of a translation, rotation, and uniform scaling.
"""

'''
# Dimensionality Reduction with PCA
'''
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)

X.shape

X_pca.shape

X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal')

"""
The Orange is the PCA
The light blue is the original data
"""