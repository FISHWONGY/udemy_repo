import numpy as np

"""
# Dimensionality Reduction

## Principal Component Analysis (PCA)
  * Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
  * [Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)
  * Statistical procedure that utilise [orthogonal transformation](https://en.wikipedia.org/wiki/Orthogonal_transformation) technology
  * Convert possible correlated features (predictors) into linearly uncorrelated features (predictors) called **principal components**
  *  of principal components <= number of features (predictors)
  * First principal component explains the largest possible variance
  * Each subsequent component has the highest variance subject to the restriction that it must be orthogonal to the preceding components. 
  * A collection of the components are called vectors.
  * Sensitive to scaling
  * [Sebastian Raschka](http://sebastianraschka.com/Articles/2014_python_lda.html): Component axes that maximise the variance

## Linear Discriminant Analysis (LDA) 
  * [Wikipedia](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
  * [Sebastian Raschka](http://sebastianraschka.com/Articles/2014_python_lda.html)
  * Most commonly used as dimensionality reduction technique in the pre-processing step for pattern-classification and machine learning applications. 
  * Goal is to project a dataset onto a lower-dimensional space with good class-separability in order avoid overfitting (“curse of dimensionality”) and also reduce computational costs.
  * Locate the 'boundaries' around clusters of classes.  
  * Projects data points on a line
  * A centroid will be allocated to each cluster or have a centroid nearby
  * [Sebastian Raschka](http://sebastianraschka.com/Articles/2014_python_lda.html): Maximising the component axes for class-separation

### Other Dimensionality Reduction Techniques

* [Multidimensional Scaling (MDS) ](http://scikit-learn.org/stable/modules/manifold.html#multi-dimensional-scaling-mds)
  * Seeks a low-dimensional representation of the data in which the distances respect well the distances in the original high-dimensional space.


* [Isomap (Isometric Mapping)](http://scikit-learn.org/stable/modules/manifold.html#isomap)

  * Seeks a lower-dimensional embedding which maintains geodesic distances between all points.


* [t-distributed Stochastic Neighbor Embedding (t-SNE)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)

  * Nonlinear dimensionality reduction technique that is particularly well-suited for embedding high-dimensional data into a space of two or three dimensions, which can then be visualized in a scatter plot. 
  * Models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points. dimensional space (e.g., to visualize the MNIST images in 2D).

***
# Gentle introduction to Linear Algebra

Linear Algebra revision:

$$A=\begin{bmatrix} 1. & 2. \\ 10. & 20. \end{bmatrix}$$

$$B=\begin{bmatrix} 1. & 2. \\ 100. & 200. \end{bmatrix}$$

\begin{align}
A \times B & = \begin{bmatrix} 1. & 2. \\ 10. & 20. \end{bmatrix} \times \begin{bmatrix} 1. & 2. \\ 100. & 200. \end{bmatrix} \\
& = \begin{bmatrix} 201. & 402. \\ 2010. & 4020. \end{bmatrix} \\
\end{align}

By parts:
$$A \times B = \begin{bmatrix} 1. \times 1. + 2.  \times 100. &  1. \times 2. + 2. \times 200. \\ 
10. \times 1. + 20. \times 100. & 10. \times 2. + 20. \times 200. \end{bmatrix}$$


"""

A = [[1., 2.], [10., 20.]]
B = [[1., 2.], [100., 200.]]

A

B

np.dot(A, B)

