# 100 Days Of ML - LOG

## Day 0 : July 17 , 2018
 
**Today's Progress** : Read though the following algorithms  
* ALFS - Active Learning and Feature Selection
* CUR decomposition
* LIME - Locally Interpretable Model Agnostic Explanations

**Thoughts** : 
* Singular-Value-Decomposition (SVD) provides the best matrix decomposition for computing a given rank-k approximation.  However, because the SVD provides these approximating basis vectors as linear combinations of the other matrix row and column vectors, it thereby fails to lead to an intuitive understanding of just which actual rows and columns, i.e. which data instances and which features, are dominant in the data matrix X. CUR Decomposition, is based on a matrix factorization method which does not provide quite as accurate an approximation to the original data set, but does so using an actual subset of rows and columns from the original data matrix X. Here, the C matrix is composed of a reduced set of actual columns from the data matrix X, and the R matrix is composed of a reduced subset of actual row vectors from X.   

* ALFS assumes data matrix X is not annotated. 
  * It tries to find those data rows which, if annotated, would provide the most value for subsequent input into a classifier.  This is called the Active-Learning step.
  * Moreover, it also simultaneously tries to find those feature columns which would be most influential as inputs for classification performance. This is called the Feature-Selection step.  

* Lime talks about how to know if the predictions made by a moddel can be trusted or are reasonable. To be model agnostic, lime can't peek into the model.
 * Local refers to local fidelity - i.e., we want the explanation to really reflect the behaviour of the classifier "around" the instance being predicted.
 * Explains those classifiers in terms of interpretable representations (words), even if that is not the representation actually used by the classifier which are not too long
 * Perturb the input around the neighborhood to see how the predictions change
 * Weight these perturbed data points by their proximity to the original example

**Link to Work:** 


## Day 1 : July 18 , 2018

**Today's Progress** :


**Thoughts** : 

**Link to Work:** 

## Day 2 : July 19 , 2018

**Today's Progress** :

**Thoughts** : 

**Link to Work:** 

## Day 3 : July 20 , 2018

**Today's Progress** :

**Thoughts** : 

**Link to Work:** 

## Day 0 : July 18 , 2018

**Today's Progress** :

**Thoughts** : 

**Link to Work:** 
