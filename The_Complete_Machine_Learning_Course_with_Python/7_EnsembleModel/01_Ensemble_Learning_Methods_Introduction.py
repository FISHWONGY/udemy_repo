import pandas as pd

"""
Ensemble Learning Methods Introduction

Supervised Learning

Group of predictors (classifiers / regressors)

## Ensemble Methods:

### **Bootstrap Aggregating** or [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
* [Scikit- Learn Reference](http://scikit-learn.org/stable/modules/ensemble.html#bagging)
* Bootstrap sampling: Sampling with replacement
* Combine by averaging the output (regression)
* Combine by voting (classification)
* Can be applied to many classifiers which includes ANN, CART, etc.

### [Pasting](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
* Sampling without replacement

### [Boosting](https://en.wikipedia.org/wiki/Boosting_(machine_learning)
* Train weak classifiers 
* Add them to a final strong classifier by weighting. Weighting by accuracy (typically)
* Once added, the data are reweighted
  * Misclassified samples gain weight 
  * Correctly classified samples lose weight (Exception: Boost by majority and BrownBoost - decrease the weight of repeatedly misclassified examples). 
  * Algo are forced to learn more from misclassified samples
  
    
### [Stacking](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)
* Also known as Stacked generalization
* [From Kaggle:](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/) Combine information from multiple predictive models to generate a new model. Often times the stacked model (also called 2nd-level model) will outperform each of the individual models due its smoothing nature and ability to highlight each base model where it performs best and discredit each base model where it performs poorly. For this reason, stacking is most effective when the base models are significantly different. 
* Training a learning algorithm to combine the predictions of several other learning algorithms. 
  * Step 1: Train learning algo
  * Step 2: Combiner algo is trained using algo predictions from step 1.

### Other Ensemble Methods:

[Wikipedia](https://en.wikipedia.org/wiki/Ensemble_learning)
* Bayes optimal classifier
  * An ensemble of all the hypotheses in the hypothesis space. 
  * Each hypothesis is given a vote proportional to the likelihood that the training dataset would be sampled from a system if that hypothesis were true. 
  * To facilitate training data of finite size, the vote of each hypothesis is also multiplied by the prior probability of that hypothesis. 
* Bayesian parameter averaging
  * an ensemble technique that seeks to approximate the Bayes Optimal Classifier by sampling hypotheses from the hypothesis space, and combining them using Bayes' law.
  * Unlike the Bayes optimal classifier, Bayesian model averaging (BMA) can be practically implemented. 
  * Hypotheses are typically sampled using a Monte Carlo sampling technique such as MCMC. 
* Bayesian model combination
  * Instead of sampling each model in the ensemble individually, it samples from the space of possible ensembles (with model weightings drawn randomly from a Dirichlet distribution having uniform parameters). 
  * This modification overcomes the tendency of BMA to converge toward giving all of the weight to a single model. 
  * Although BMC is somewhat more computationally expensive than BMA, it tends to yield dramatically better results. The results from BMC have been shown to be better on average (with statistical significance) than BMA, and bagging.
* Bucket of models
  * An ensemble technique in which a model selection algorithm is used to choose the best model for each problem. 
  * When tested with only one problem, a bucket of models can produce no better results than the best model in the set, but when evaluated across many problems, it will typically produce much better results, on average, than any model in the set.


R released
* BMS (an acronym for Bayesian Model Selection) package
* BAS (an acronym for Bayesian Adaptive Sampling) package
* BMA package

**Note: Ensemble methods**

* Work best with indepedent predictors

* Best to utilise different algorithms

***
"""