import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

# !wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
df = pd.read_csv('/Users/yuawong/Documents/GitHub/udemy_repo/The_Complete_Machine_Learning_Course_with_Python/'
                 'data/wine.data', header=None)
df.shape

df.head()

col_name = ['class', 'Alcohol', 'Malic acid', 'Ash',	'Alcalinity of ash',
            'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
            'Proanthocyanins', 'Color intensity', 'Hue',
            'OD280/OD315 of diluted wines', 'Proline']

df.columns = col_name

X = df.iloc[:, 1:]
X.head()

y = df['class']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

sc.fit(X_train)
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X_train_sc)

pca.explained_variance_ratio_

print(np.round(pca.explained_variance_ratio_, 3))
# 1st pca component explain 36% of data, the 2nd explained 18.7%

pd.DataFrame(np.round(pca.components_, 3), columns=X.columns).T
# 0 = PCA component 1, 1 = PCA component 2

"""No preconceived idea of the number of PCAs we want"""

pca = PCA(n_components=None)
pca.fit(X_train_sc)

pca.transform(X_train_sc)

print(np.round(pca.explained_variance_ratio_, 3))

np.cumsum(pca.explained_variance_ratio_)

plt.plot(np.cumsum(pca.explained_variance_ratio_)*100.)
plt.xlabel('number of components')
plt.ylabel('cummulative explained variance');
# If want to have 80% of variance to be explained, 4 component required, if 90%, 6 component

res = pca.transform(X_train_sc)
index_name = ['PCA_'+str(k) for k in range(0, len(res))]

df1 = pd.DataFrame(res, columns=df.columns[1:],
                   index=index_name)[0:4]
df1.T.sort_values(by='PCA_0')

