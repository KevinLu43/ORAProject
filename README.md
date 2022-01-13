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
  <img src="https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_PIC_1.jpg">
</p>
The Fig.1 demonstrates the flow chart of Generalized Ensemble Method (GEM) and Generalized Weighted Ensemble with Internally Tuned Hyperparameters (GEM–ITH).The GEM was proposed to find the best combination of base learners by Perrone and Cooper (1992). It minimizes the total  mean square error (MSE) to optimize the weight of each base learner. The fomulation is shown as bellow:

![image](https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_PIC_2.JPG)

The above formulation is a nonlinear convex optimization problem. As the constraints are linear, computing the Hessian matrix will demonstrate the convexity of the objective function. Hence, since a local optimum of a convex function (objective function) on a convex feasible region (feasible region of the above formulation) is guaranteed to be a global optimum, the optimal solution of this problem is proved to be the global optimal solution (Boyd and Vandenberghe, 2004).

The GEM-ITH algorithm is using  a heuristic based on Bayesian search that aims at finding some candidate hyperparameter values for each base learner and obtain the best weights and hyperparameters combination for the ensemble of all base models. Given *n* iterations of Bayesian optimization, the hyper parameter and combinations for each base learner that has the best performance among the iterations was considered as the optimal combination by GEM-ITH model.

The origin GEM assumes hyper parameters of each base learner  are tuned with one of the many common tuning approaches before conducting the ensemble weighting task. We use the random search as the tuning approach in this project.

On the other hand, the tuning and optimize procedure may need huge computational resource, we take a two stage GEM-ITH model to prevent reduce the computational resource and compare the performance between the origin GEM-ITH medol and two stage GEM-ITH medol. The two stage GEM-ITH model is shown as the figure bellow.

<p align="center">
  <img src="https://github.com/KevinLu43/ORAProject/blob/main/picture/ORA_PIC_GEMITH2.PNG">
</p>

The two stage GEM-ITH

Data Collection and Analysis Result
---

Conclusion
---

Feature Work 
---
Tran et al. (2020) had proposed a framework of 

---

Reference
---
* Boyd, S., & Vandenberghe, L. Convex optimization: Cambridge university press, 2004.
* Hastie, T., Tibshirani, R., Friedman, J., & Franklin, J. The elements of statistical learning: data mining, inference and prediction. The Mathematical Intelligencer, 27(2), 83-85, 2005.
* Lu, H. W., & Lee, C. Y. Kernel-Based Dynamic Ensemble Technique for Remaining Useful Life Prediction. IEEE Robotics and Automation Letters, 2021.
* Perrone, M. P., & Cooper, L. N. When networks disagree: Ensemble methods for hybrid neural networks: BROWN UNIV PROVIDENCE RI INST FOR BRAIN AND NEURAL SYSTEMS, 1992.
* Shahhosseini, M., Hu, G., & Pham, H. Optimizing ensemble weights and hyperparameters of machine learning models for regression problems. arXiv preprint arXiv:1908.05287, 2019.
* Tran, N., Schneider, J. G., Weber, I., & Qin, A. K. Hyper-parameter optimization in classification: To-do or not-to-do. Pattern Recognition, 103, 107245, 2020.
* Yang, L., & Shami, A. On hyperparameter optimization of machine learning algorithms: Theory and practice. Neurocomputing, 415, 295-316, 2020.

