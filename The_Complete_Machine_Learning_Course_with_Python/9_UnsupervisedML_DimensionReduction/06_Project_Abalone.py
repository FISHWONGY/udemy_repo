# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

"""# [Abalone](http://archive.ics.uci.edu/ml/datasets/Abalone)

8 features

Applying PCA. PCA - 80% of variance.
"""

# !wget http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data
df1 = pd.read_csv('abalone.data', header=None)
df1.head()

X = df1.iloc[:, 1:]

X.head()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

from sklearn.decomposition import PCA

pca = PCA(n_components=None)
X_sc = sc.fit_transform(X)
pca.fit(X_sc)
np.cumsum(pca.explained_variance_ratio_)

plt.plot(np.cumsum(pca.explained_variance_ratio_)*100.)
plt.xlabel('number of components')
plt.ylabel('cummulative explained variance')

