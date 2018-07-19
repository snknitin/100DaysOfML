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
* Looked at **Uber's Michealangelo**. It is designed to cover the end-to-end ML workflow: manage data, train, evaluate, and deploy models, make predictions, and monitor predictions. The system also supports traditional ML models, time series forecasting, and deep learning. UberEATS has several models running on Michelangelo, covering meal delivery time predictions, search rankings, search autocomplete, and restaurant rankings. The delivery time models predict how much time a meal will take to prepare and deliver before the order is issued and then again at each stage of the delivery process.
* Read through the **RL^2 paper for faster RL agorithm training using RNN**


**Thoughts** : 
* Michealangelo is an internal ML-as-a-service platform that democratizes machine learning and makes scaling AI to meet the needs of business as easy as requesting a ride.The focus shifted to developer productivity–how to speed up the path from idea to first production model and the fast iterations that follow. 
   * Predicting meal estimated time of delivery (ETD) is not simple. When an UberEATS customer places an order it is sent to the restaurant for processing. The restaurant then needs to acknowledge the order and prepare the meal which will take time depending on the complexity of the order and how busy the restaurant is. When the meal is close to being ready, an Uber delivery-partner is dispatched to pick up the meal. Then, the delivery-partner needs to get to the restaurant, find parking, walk inside to get the food, then walk back to the car, drive to the customer’s location (which depends on route, traffic, and other factors), find parking, and walk to the customer’s door to complete the delivery. 
   * The goal is to predict the total duration of this complex multi-stage process, as well as recalculate these time-to-delivery predictions at every step of the process.
   * They use gradient boosted decision tree regression models to predict this end-to-end delivery time.
   * Features for the model include information from the request (e.g., time of day, delivery location), historical features (e.g. average meal prep time for the last seven days), and near-realtime calculated features (e.g., average meal prep time for the last one hour).
   * The primary open sourced components used are HDFS, Spark, Samza, Cassandra, MLLib, XGBoost, and TensorFlow.
   *  To provide scalable, reliable, reproducible, easy-to-use, and automated tools to address the following six-step workflow:
      * Manage data - building and managing data pipelines is typically one of the most costly pieces of a complete machine learning solution.
          * A platform should provide standard tools for building data pipelines to generate feature and label data sets for training (and re-training) and feature-only data sets for predicting
          * need to be scalable and performant,  incorporate integrated monitoring for data flow and data quality, and support both online and offline training and predicting.
          * Ideally, they should also generate the features in a way that is shareable across teams to reduce duplicate work and increase data quality.
          * The offline pipelines are used to feed batch model training and batch prediction jobs and the online pipelines feed online, low latency predictions 
      * Train models
      * Evaluate models
      * Deploy models
      * Make predictions
      * Monitor predictions



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
