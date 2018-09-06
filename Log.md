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

to be continued ....


**Link to Work:** 

## Day 2 : July 19 , 2018

**Today's Progress** : * Read through the **RL^2 paper for faster RL agorithm training using RNN** to ask Pieter Abbeel some doubts during our meeting

**Thoughts** : 

1) In visual experiments, the algorithm/agent interacts with maze for multiple episodes during which the maze and target position are held fixed. Is our target going to be finding the bug in the verilog code? If so, we don't have a fixed location(target) ?

2) First we explore and identify target and then act optimally. How do we make it explore efficiently with some shaped rewards.? Do we add some exploration bonus for new conditions that are infrequently explored.?would that undermine this one shot learning process if our problem statement is more likely to have localized issues in the nested if else for several problems. Is there a way to direct the exploration in a hierarchical setting that is not fixed in structure? Perhaps using rnn to identify test cases or words in the conditions.

3) Would the exploitation strategy be thrown off by masking bugs that undermine other bugs? Should we treat our environment to be stateless without dependencies of the nested loops. Is multi arm bandit  a correct analogy for us?

4) Since we have a hierarchy in the given nested if else problem but it could be of a dynamic nature in length, so should we treat this as a seq2seq where we encode and decode the conditional statement into a tuple of state, action, observation, reward?

5) Which rl algorithm would suit our need to make use of the prior information, for the outer loop?

**Link to Work:** 

## Day 3 : July 20 , 2018

**Today's Progress** : Deep RL bootcamp lectures 1 and 2

**Thoughts** : Introductory MDP stuff and Sample approximation with function fitting


**Link to Work:** 

## Day 4 : July 21 , 2018

**Today's Progress** : Deep RL bootcamp lectures 3 and 4

**Thoughts** : 

**Link to Work:** 


## Day 5 : July 22 , 2018

**Today's Progress** : Deep RL bootcamp lectures 5 and 6

**Thoughts** : 

**Link to Work:** 


## Day 6 : July 23 , 2018

**Today's Progress** : Deep RL bootcamp lectures 7 and 8

**Thoughts** : 

**Link to Work:** 


## Day 7 : July 24 , 2018

**Today's Progress** : IPython Magic Commands

**Thoughts** : Reference notecook for some useful commands

**Link to Work:** 

* http://nbviewer.jupyter.org/github/jdwittenauer/ipython-notebooks/blob/master/notebooks/language/IPythonMagic.ipynb


## Day 8 : July 25 , 2018

**Today's Progress** : Features selected on the basis of their scores in various statistical tests for their correlation with the outcome variable. 
* Pearson Correlation - It is used as a measure for quantifying linear dependence between two continuous variables X and Y. Its value varies from -1 to +1. 
* LDA - Linear Discriminant Analysis -  used to find a linear combination of features that characterizes or separates two or more classes (or levels) of a categorical variable.
* ANOVA - Analysis of Variance. similar to LDA except for the fact that it is operated using one or more categorical independent features and one continuous dependent feature. It provides a statistical test of whether the means of several groups are equal or not.
* Chi-square : statistical test applied to the groups of categorical features to evaluate the likelihood of correlation or association between them using their frequency distribution.

**Thoughts** : Top reasons to use feature selection are:

It enables the machine learning algorithm to train faster.  
It reduces the complexity of a model and makes it easier to interpret.  
It improves the accuracy of a model if the right subset is chosen.  
It reduces overfitting.  



**Link to Work:**  

https://towardsdatascience.com/why-how-and-when-to-apply-feature-selection-e9c69adfabf2
https://machinelearningmastery.com/feature-selection-machine-learning-python/


## Day 9-10 : August 1-2 , 2018

**Today's Progress** : Using Seaborn to visualize and plot graphs for feature engineering
* distplot(df["feature_name"], label="") To get distribution of data values
* countplot(df["feature_name"], label="") To get count of categorical values
* barplot(x= "feature_name", y= "Class", data=df) To get bar plot of featuere values against true classification label

**Thoughts** :

* Look at featuere values and the number of unique values
* Divide the value counts by the training data size and see what percentage of the data the top five values capture
* If majority(more than 85%) of that feature value is 0, then it is probably useless
* If the feature has way too many unique categorical value, it would add a lot of dimensionality and is probably not requrired
* If a feature has the same value through out, then it gives no additional information
* If it is skewed, look at the distribution in a specific range and decide whether you need it
* Map the categorical values into one hot encodings



**Link to Work:**  

https://github.com/snknitin/Tf-skeleton/blob/master/MultiClassification.ipynb


## Day 11-14 : August 3-6 , 2018

**Today's Progress** : Used keras to create a DNN architecture to predict classes in loan data from kaggle

**Thoughts** :

* Tweaked different parameters and hidden layer sizes
* Looked at misclassification counts and plotted a confusion matrix

       sns.heatmap(cf_mat_p, annot=True, linewidths=.5, cmap=cm.summer,xticklabels=['bad','safe'], yticklabels=['bad','safe'])



**Link to Work:**  

https://github.com/snknitin/Tf-skeleton/blob/master/MultiClassification.ipynb


## Day 15 : August 8 , 2018

**Today's Progress** : Read the rules of ML 
**Thoughts** :
**Link to Work:**  
http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf


## Day 16 : August 9 , 2018

**Today's Progress** : Feature Scaling with scikit-learn 
**Thoughts** :

    num_cols = df.columns[df.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
 Avoid scaling boolean features
    
    bool_cols = [col for col in df if 
               df[col].dropna().value_counts().index.isin([0,1]).all()]
    
The LabelBinarizer is the right choice for encoding string columns - it will first translate strings to integers, and then binarizes those integers to bit vectors. It does it all in one go. No need to divide this "workflow" between two transformer steps (ie. LabelEncoder plus OneHotEncoder)

Check out the sklearn_pandas.DataFrameMapper meta-transformer. Use it as the first step in your pipeline to perform column-wise data engineering operations:

    mapper = DataFrameMapper(
      [(continuous_col, StandardScaler()) for continuous_col in continuous_cols] +
      [(categorical_col, LabelBinarizer()) for categorical_col in categorical_cols]
    )
    pipeline = Pipeline(
      ("mapper", mapper)
    )
    pipeline.fit_transform(df, df["y"])

**Link to Work:**  
http://benalexkeen.com/feature-scaling-with-scikit-learn/





## Day 17 : August 10 , 2018

**Today's Progress** : More Feature Engineering
**Thoughts** :

    pandas.get_dummies()
    sklearn.preprocessing.LabelEncoder()
    sklearn.preprocessing.LabelBinarizer
    sklearn.preprocessing.MultiLabelBinarizer

**Link to Work:**  

https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b
https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63



## Day 18 : August 11 , 2018

**Today's Progress** : Dealing with categorical variables
**Thoughts** :


**Link to Work:**  

http://pbpython.com/categorical-encoding.html
https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512


## Day 19 : August 13 , 2018

**Today's Progress** : Call backs to print metrics on epoch end in keras
**Thoughts** : Sometimes there is a need to see how the loss and accuracy proceed through the epochs. it would be great to print them out to console to see them

    import numpy as np
    from keras.callbacks import Callback
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    class Metrics(Callback):
        def on_train_begin(self, logs={}):
           self.val_f1s = []
           self.val_recalls = []
           self.val_precisions = []

        def on_epoch_end(self, epoch, logs={}):
           val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
           val_targ = self.model.validation_data[1]
           _val_f1 = f1_score(val_targ, val_predict)
           _val_recall = recall_score(val_targ, val_predict)
           _val_precision = precision_score(val_targ, val_predict)
           self.val_f1s.append(_val_f1)
           self.val_recalls.append(_val_recall)
           self.val_precisions.append(_val_precision)
           print “ — val_f1: %f — val_precision: %f — val_recall %f” %(_val_f1, _val_precision, _val_recall)
           return

    metrics = Metrics()

Use it like this

    model.fit(training_data, training_target, validation_data=(validation_data, validation_target), nb_epoch=10, batch_size=64, callbacks=[metrics])

Or even try :

    checkpoints =ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                    save_best_only=False, save_weights_only=False,
                                    mode='auto', period=2)

    ###############Fit Model #############################

    model.fit_generator(
    train_generator,
    steps_per_epoch =total_samples//batch_size,
    epochs = epochs,
    validation_data=validation_generator,
    validation_steps=total_validation//batch_size,
    callbacks = [checkpoints],
    shuffle= True)

**Link to Work:**  
* https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2  
* https://faroit.github.io/keras-docs/1.2.0/callbacks/



## Day 20 : August 14 , 2018

**Today's Progress** : Tensorflow Early stopping callbacks and print after epoch  
**Thoughts** :

**Link to Work:**  

http://mckinziebrandon.me/TensorflowNotebooks/2016/11/20/early-stopping.html




## Day 21 : August 15 , 2018

**Today's Progress** : DeepReplay is a nice Python package that allows you to replay the process of training your model and create nice visualizations out of it. The best thing is that it can be very easily integrated with Keras since we can just use the callback for this.

**Thoughts** : 

Deep Learning is all about hyper-parameters! The actual role of the non-linearity is to twist and turn the feature space so much so that the boundary turns out to be… LINEAR!

The process is simple enough, consisting of five steps:

* It all starts with creating an instance of a callback!

      from deepreplay.callbacks import ReplayData
      from deepreplay.datasets.parabola import load_data

      X, y = load_data()

      replaydata = ReplayData(X, y, filename='hyperparms_in_action.h5', group_name='part1')
 you can use the same file several times over, but never the same group

* Then, business as usual: build and train your model.

      model.fit(X, y, epochs=150, batch_size=16, callbacks=[replaydata])
     

* Next, load the collected data into Replay. 
Create an instance of Replay, providing the filename and the group name you chose in Step 1

    from deepreplay.replay import Replay

    replay = Replay(replay_filename='hyperparms_in_action.h5', group_name='part1')


* Finally, create a figure and attach the visualizations from your Replay object to the figure.

      import matplotlib.pyplot as plt
      fig, ax = plt.subplots(1, 1, figsize=(5, 5))
      fs = replay.build_feature_space(ax, layer_name='hidden')
      
  There are 5 visualizations available - Feature Space, Decision Boundary, Probability Histogram, Loss and Metric, Loss Histogram

* Plot and/or animate it!


      # Plot 60th epoch and save it as PNG
      fs.plot(epoch=60).savefig('feature_space_epoch60.png', dpi=120)

      # Animate and save it as MP4
      fs.animate().save('feature_space_animation.mp4', dpi=120, fps=5)
      
   If you decide to go with multiple simultaneous visualizations, there are two helper methods that return composed plots and animations, respectively: compose_plots and compose_animations.
   
      fig = plt.figure(figsize=(12, 6))
      ax_fs = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
      ax_ph_neg = plt.subplot2grid((2, 4), (0, 2))
      ax_ph_pos = plt.subplot2grid((2, 4), (1, 2))
      ax_lm = plt.subplot2grid((2, 4), (0, 3))
      ax_lh = plt.subplot2grid((2, 4), (1, 3))

      fs = replay.build_feature_space(ax_fs, layer_name='hidden')
      ph = replay.build_probability_histogram(ax_ph_neg, ax_ph_pos)
      lh = replay.build_loss_histogram(ax_lh)
      lm = replay.build_loss_and_metric(ax_lm, 'acc')

      sample_figure = compose_plots([fs, ph, lm, lh], 80)
      sample_figure.savefig('part1.png', dpi=120, format='png')

      sample_anim = compose_animations([fs, ph, lm, lh])
      sample_anim.save(filename='part1.mp4', dpi=120, fps=5)


**Link to Work:**  
* https://towardsdatascience.com/hyper-parameters-in-action-a524bf5bf1c
* https://towardsdatascience.com/hyper-parameters-in-action-introducing-deepreplay-31132a7b9631
* https://github.com/dvgodoy/deepreplay


## Day 22 - 36 : August 16 - 29 , 2018

**Today's Progress** :  Introduction to Manual and Automated Feature Engineering  
**Thoughts** :  

* To determine if a variable is useful, we can calculate the Pearson Correlation Coefficient (r-value) between this variable and the target. This measures the strength of a linear relationship between two variables and ranges from -1 (perfectly negatively linear) to +1 (perfectly positively linear). The r-value is not best measure of the "usefulness" of a new variable, but it can give a first approximation of whether a variable will be helpful to a machine learning model. The larger the r-value of a variable with respect to the target, the more a change in this variable is likely to affect the value of the target. Therefore, we look for the variables with the greatest absolute value r-value relative to the target.

We can also visually inspect a relationship with the target using the Kernel Density Estimate (KDE) plot

    # Calculate the correlation coefficient between the new variable and the target
    corr = df[label_name].corr(df[var_name])
    sns.kdeplot(df.ix[df[label_name] == value, var_name], label = 'Label : '+ str(value))

* Pandas tricks 

      df1.merge(df2, on = 'SK_ID_CURR', how = 'left')   # Merging or joins
      categorical = pd.get_dummies(df.select_dtypes('object'))  # Categorical columns


**Link to Work:** 
* https://www.featurelabs.com/blog/secret-to-data-science-success/
* http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
* https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering
* https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering-p2
* https://www.kaggle.com/willkoehrsen/clean-manual-feature-engineering
* https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics
* https://stackoverflow.com/questions/46498455/categorical-features-correlation
* https://www.jeremyjordan.me/preparing-data-for-a-machine-learning-model/
* https://www.kaggle.com/danavg/dummy-variables-vs-label-encoding-approach
* https://www.analyticsvidhya.com/blog/2015/11/easy-methods-deal-categorical-variables-predictive-modeling/
* https://stackoverflow.com/questions/20892799/using-pandas-calculate-cramérs-coefficient-matrix
* https://stackoverflow.com/questions/46498455/categorical-features-correlation
* https://medium.com/ml-byte/rare-feature-engineering-techniques-for-machine-learning-competitions-de36c7bb418f
* https://realpython.com/python-pandas-tricks/


## Day 37 : August 30 , 2018


**Today's Progress** :  When to Use MLP, CNN, and RNN Neural Networks  
**Thoughts** :   

* **DNN** - They are very flexible and can be used generally to learn a mapping from inputs to outputs. For example, the pixels of an image can be reduced down to one long row of data and fed into a MLP. The words of a document can also be reduced to one long row of data and fed to a MLP. Even the lag observations for a time series prediction problem can be reduced to a long row of data and fed to a MLP.  
  * Use MLPs For:
    * Tabular dataset
    * Classification prediction problems
    * Regression prediction problems  
  * Try MLPs On:
    * Image data
    * Text Data
    * Time series data
    * Other types of data
* **CNN** - Convolutional Neural Networks, or CNNs, were designed to map image data to an output variable. More generally, CNNs work well with data that has a spatial relationship.The CNN input is traditionally two-dimensional, a field or matrix, but can also be changed to be one-dimensional, allowing it to develop an internal representation of a one-dimensional sequence.Although not specifically developed for non-image data, CNNs achieve state-of-the-art results on problems such as document classification used in sentiment analysis and related problems
  * Use CNNs For:
    * Image data
    * Classification prediction problems
    * Regression prediction problems  
  * Try CNNs On:
    * Text Data
    * Time series data
    * Sequence Input data
* **RNN** - Recurrent Neural Networks, or RNNs, were designed to work with sequence prediction problems like 1-1, 1-many, many-1, many-many(seq2seq).The Long Short-Term Memory, or LSTM, network is perhaps the most successful RNN because it overcomes the problems of training a recurrent network and in turn has been used on a wide range of applications
  * Use RNNs For:
    * Speech data
    * Text Data
    * Generative models
    * Classification prediction problems
    * Regression prediction problems  
  * DON'T Try RNNs On:
    * Image Data
    * Tabular data

**Link to Work:**    
* https://machinelearningmastery.com/when-to-use-mlp-cnn-and-rnn-neural-networks/


## Day 38 : August 31 , 2018


**Today's Progress** :  AutoML and AutoKeras    
**Thoughts** :   
**Link to Work:**    
* https://www.automl.org
* https://www.kdnuggets.com/2018/08/auto-keras-create-deep-learning-model-4-lines-code.html
* https://towardsdatascience.com/autokeras-the-killer-of-googles-automl-9e84c552a319


## Day 39 : September 1 , 2018

**Today's Progress** :  Using fastText and Comet.ml to classify relationships in Knowledge Graphs   
**Thoughts** :   
**Link to Work:**   
* https://medium.com/comet-ml/using-fasttext-and-comet-ml-to-classify-relationships-in-knowledge-graphs-e73d27b40d67



## Day 39 : September 2 , 2018

**Today's Progress** : Multi-Modal Few-Shot Learning: A Benchmark , ACTIVE Learning   
**Thoughts** :   
*  Deep learning algorithms generally require large amounts of data for model training and a multi-modal approach facilitates bridging the information gap by means of meaningful joint embeddings.
*  Multi-modal during training (i.e. images and texts) and single-modal in testing time (i.e. images), with the associated task to utilize multi-modal data in base classes (with many samples), to learn explicit visual classifiers for novel classes (with few samples)

ACTIVE LEARNING


Typical heuristics for active learning:
	- Start with a pool of unlabeled data
	- Pick a few points at random and get their labels
	- Repeat
		- Fit a classifier to the labels seen so far
		- Query the unlabeled point that is closest to the boundary
		(or most uncertain, or most likely to decrease overall
		uncertainty,...)

Biased sampling: the labeled points are not representative of the underlying distribution!

There are two distinct narratives for explaining how adaptive querying can help.  
Case I: Exploiting (cluster) structure in data  

Challenges:  
In general, the cluster structure  
(i) is not so clearly defined and  
(ii) exists at many levels of granularity. And the clusters themselves might not be pure in their labels. How to exploit whatever structure happens to exist?


- Build neighborhood graph
- Query some random points
- Propagate labels
- Make query and go to (3)

- Use hierarchical clustering


Case II: Efficient search through hypothesis space  

Ideal case: each query cuts the version space in two. Then perhaps we need just log |H| labels to get a perfect hypothesis!


**Link to Work:**  
* https://openreview.net/pdf?id=HJB8ntJPG
* http://hunch.net/~active_learning/active_learning_icml09.pdf
* https://www.datacamp.com/community/tutorials/active-learning




## Day 40-42 : September 3 -4 , 2018

**Today's Progress** :  Clustering Analysis  
**Thoughts** :   
**Link to Work:**   
* https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68
* https://github.com/snknitin/Cluster-Analysis

## Day 43 : September 5 , 2018

**Today's Progress** :  How to make animated gifs from Matplotlib plots easily  
**Thoughts** :   Using Facebook data to plot my friend network
**Link to Work:**   
https://ndres.me


## Day 44 : September 6  , 2018

**Today's Progress** :  Image-to-image translation in the browser  
**Thoughts** :  
**Link to Work:**   
* https://github.com/zaidalyafeai/zaidalyafeai.github.io/tree/master/pix2pix
* https://www.linkedin.com/feed/update/urn:li:activity:6435094001153507328/


## Day 45 : September 7  , 2018
 
**Today's Progress** : Deep Learning Reproducibility     
**Thoughts** :   
**Link to Work:**    
https://www.youtube.com/watch?v=Ys8ofBeR2kA


## Day 46 : September 8 , 2018
 
**Today's Progress** : Montreal AI course     
**Thoughts** :   
**Link to Work:**    
* https://montrealartificialintelligence.com/academy/#Getting-Started-Readings-Source-Code-and-Science

## Day 47 : September 9  , 2018

**Today's Progress** :  Altair : Declarative statistical visualization library for Python   
**Thoughts** :  
**Link to Work:**   
* https://github.com/altair-viz/altair


## Day 48 : September 10 , 2018

**Today's Progress** :  NLP with Python
**Thoughts** :  
**Link to Work:**   
* www.linkedin.com/learning/nlp-with-python-for-machine-learning-essential-training

## Day 49 : September 11 , 2018

**Today's Progress** :  Flappy Bird in PyTorch RL
**Thoughts** :  
**Link to Work:**   
* https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial


## Day 50 :September 12 , 2018

**Today's Progress** :  Exploring the top 20 data science related libraries for python
**Thoughts** :  
**Link to Work:**   
* https://www.kdnuggets.com/2018/06/top-20-python-libraries-data-science-2018.html


## Day 51 : September 13 , 2018

**Today's Progress** :  Keras Deep Learning REST API
**Thoughts** :  
**Link to Work:**   
* https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html?utm_content=buffer08c4c&utm_medium=social&utm_source=linkedin.com&utm_campaign=buffer
