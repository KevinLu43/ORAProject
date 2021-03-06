Optimizing Ensemble Weights and Hyper Parameters of Machine Learning Models
===

This project is for Operations Research Applications and Implementation, IMIS, NCKU.

Background and Motivation
---
Nowadays, machine learning has become popular in every area. There always are novel methods been proposed, however, there usually are some parameters or hyper parameters need to be set in order to construct the model. The suitable hyper parameters for the model may affect the performance of the model.

On the other hand, aggregating multiple models which were called ensemble method also been widely used in recent years.Basic ensemble method combines several regression base learners by averaging their estimates. Another ensemble method uses the linear combination of the regression base learners. The weights are assigned to each of the base learners according to their performance on the validation set so called weighted majority approach is the other idea to integrate all predictions.

Therefore, how to assign the proper ensemble wights and set hyper parameters of machine learning models is the topic, Shahhosseini and Phamthere (2019) proposed a framework that combine the model hyperparameter tuning and the model weights aggregation for optimal ensemble design in one coherent process, we follow the framework and try to implement the method in the remaining useful life prediction.

Problem Definition
---
As mention, the main purpose of this project can conclude as follow:
* Optimize the hyper parameters of each base model .
* Optimize the weight of the base models to improve the final predictions.
* The procedure of optimization is necessary or not.

Methodology
---
<p align="center">
  <img src="https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_PIC_GEMITH.PNG">
</p>
The Fig.1 demonstrates the flow chart of Generalized Ensemble Method (GEM) and Generalized Weighted Ensemble with Internally Tuned Hyperparameters (GEM–ITH).The GEM was proposed to find the best combination of base learners by Perrone and Cooper (1992). It minimizes the total  mean square error (MSE) to optimize the weight of each base learner. The fomulation is shown as bellow:

![image](https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_PIC_2.JPG)

The above formulation is a nonlinear convex optimization problem. As the constraints are linear, computing the Hessian matrix will demonstrate the convexity of the objective function. Hence, since a local optimum of a convex function (objective function) on a convex feasible region (feasible region of the above formulation) is guaranteed to be a global optimum, the optimal solution of this problem is proved to be the global optimal solution (Boyd and Vandenberghe, 2004).

The GEM-ITH algorithm (Shahhosseini and Pham, 2019) is using  a heuristic based on Bayesian search that aims at finding some candidate hyperparameter values for each base learner and obtain the best weights and hyperparameters combination for the ensemble of all base models. Given *n* iterations of Bayesian optimization, the hyper parameter and combinations for each base learner that has the best performance among the iterations was considered as the optimal combination by GEM-ITH model.

The origin GEM assumes hyper parameters of each base learner  are tuned with one of the many common tuning approaches before conducting the ensemble weighting task. We use the random search as the tuning approach in this project.

On the other hand, the tuning and optimize procedure may need huge computational resource, we take a two stage GEM-ITH model to prevent reduce the computational resource and compare the performance between the origin GEM-ITH medol and two stage GEM-ITH medol. The two stage GEM-ITH model is shown as the figure bellow.

<p align="center">
  <img src="https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_PIC_GEMITHS2.PNG">
</p>

The two stage GEM-ITH integrate the GEM and GEM-ITH, we tuned the hyper parameters at stage one and take the hyper parameters as the initial solution to the stage two and excate the optimize procedure.

<p align="center">
  <img src="https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_hpsetting.PNG">
</p>

Table. 1 shows the hyper parameter of machine learning model which were widely used recently and the setting of these models in GEM-ITH method. For base learner, support vector regression (SVR), random forest (RF), gradient boosting machine (GBM) are chosen for comparison.

**Data Collection and Analysis Result**
---
#### Data collection

<p align="center">
  <img src="https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_PIC_DatasetDetail.PNG">
  <img src="https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_PIC_RawData.png">
</p>

The dataset we used is an experiment of the accelerated electrical discharge destruction of rotary bearings. The dataset is collected from four accelerometers, three-phase voltage, three-phase current, a rotary encoder, and a torque meter.
The rotation rate is 1800 revolutions per minute (30 Hz). The vibration amplitude of the bearing is collected from LabVIEW program with sampling frequency 25600 data points per second (25.6kHz). The collection interval is 20 seconds for every 10 minutes. A complete experiment would last 24 to 27 hours, which implies that a complete experiment collects 73 to 83 million
data points. The detail of data collection is shown as table. 2, Fig. 3 shows the raw data. In order to estimate the RUL, the data points are too big to construct the prediction model, we use the same data preprocessing procedure as Lu and Lee (2021) and have the dataset which obtain 900 observations, 15 independent variables and 1 dependent variable. To prevent the model just fit the specific dataset, we replicate 30 times to verify the method in the project.

#### Analysis Result

<p align="center">
  <img src="https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_thebest.PNG">
</p>

Table. 3 illustrates the average performance of each model in 30 testing datasets. The GEM-ITH here is the two stage GEM-ITH model, the result shows that the GEM-ITH can have the best performance but need  a bunch of computation resource.

<p align="center">
 <img src="https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_basemodel.PNG">
 <img src="https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_ensemblemodel.PNG">
</p>

In Table. 4, we compared the improvement percentage of optimization between base learners, the results illstrate the great improvement on every metrics after tuned hyper parameters of base learners. Table. 5 compared the performance of ensemble method and the progressive of using GEM and GEM-ITH. When GEM-ITH with initial solution, the computation time have remarkable reduction.

<p align="center">
 <img src="https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_iterationvs.PNG">
 <img src="https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_frontier1.PNG">
</p>

As the previous table showed, the performance of model improved with computation resource increased. According the law of diminishing returns, the improvement will not grow without limits, we gave the different iterations to GEM-ITH, the results showed as fig. 4 and fig. 5, when iterations smaller than 100, the improvement is significant and the computation time is affordable, but the computation time need to be considered when iterations bigger than 100. The fig. 6 illustrats the frontier between computation time and improvement. The hyper parameter comparison is shown as following table, the results show that the best hyper parameter are not the same when we use different method, but the performance may not have significant difference.

<p align="center">
 <img src="https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_hyper1.png">
</p>

**Conclusion**
---

In conclusion, optimizing ensemble weights and hyper parameters can improve the performance in this case, but the optimizing process highly demanding computational cost. Therefore, we compared the computation time and the improvement, the frontier can help us to estimate the cost of computation resource whice are affordable. 
On the other hand, when give the initial solution can reduce the computation time in this case, and improve the performance, this can help us get the acceptable solution in limited computation resource.

**Feature Work**
---

Tran et al. (2020) have proposed a framework to address the problem of whether one should apply hyper parameter optimization or use the default hyper parameter settings for traditional classification algorithms, the framework is shown as following figure, the work in this project is in the red box, we are considering to follow the whole framework in the feature.

<p align="center">
 <img src="https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_feature.PNG">
</p>

Reference
---
* Boyd, S., & Vandenberghe, L. Convex optimization: Cambridge university press, 2004.
* Hastie, T., Tibshirani, R., Friedman, J., & Franklin, J. The elements of statistical learning: data mining, inference and prediction. The Mathematical Intelligencer, 27(2), 83-85, 2005.
* Lu, H. W., & Lee, C. Y. Kernel-Based Dynamic Ensemble Technique for Remaining Useful Life Prediction. IEEE Robotics and Automation Letters, 2021.
* Perrone, M. P., & Cooper, L. N. When networks disagree: Ensemble methods for hybrid neural networks: BROWN UNIV PROVIDENCE RI INST FOR BRAIN AND NEURAL SYSTEMS, 1992.
* Shahhosseini, M., Hu, G., & Pham, H. Optimizing ensemble weights and hyperparameters of machine learning models for regression problems. arXiv preprint arXiv:1908.05287, 2019.
* Tran, N., Schneider, J. G., Weber, I., & Qin, A. K. Hyper-parameter optimization in classification: To-do or not-to-do. Pattern Recognition, 103, 107245, 2020.
* Yang, L., & Shami, A. On hyperparameter optimization of machine learning algorithms: Theory and practice. Neurocomputing, 415, 295-316, 2020.

