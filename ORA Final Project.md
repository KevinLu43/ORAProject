# **ORA Final Project**
## **Optimizing Ensemble Weights and Hyper Parameters of Machine Learning Models**

This project is for Operations Research Applications and Implementation.

Background and Motivation
---
Nowadays, machine learning has become popular in every area. There always are novel methods been proposed, however, there usually are some parameters or hyper parameters need to be set in order to construct the model. The suitable hyper parameters for the model may affect the performance of the model.

On the other hand, aggregating multiple models which were called ensemble method also been widely used in recent years.Basic ensemble method combines several regression base learners by averaging their estimates. Another ensemble method uses the linear combination of the regression base learners. The weights are assigned to each of the base learners according to their performance on the validation set so called weighted majority approach is the other idea to integrate all predictions.

Therefore, how to assign the proper ensemble wights and set hyper parameters of machine learning models is the topic, Shahhosseini and Phamthere (2019) proposed a flow chart that combine the model hyperparameter tuning and the model weights aggregation for optimal ensemble design in one coherent process, we follow the flow chart and try to implement the method in the remaining useful life prediction.

Problem Definition
---
As mention, the main purpose of this project can conclude as follow:
* Optimize the hyper parameters of each base model .
* Optimize the weight of the base models to improve the final predictions.
* The procedure of optimization is necessary or not.

Methodology
---

The Fig.1 demonstrates the flow chart of Generalized Ensemble Method (GEM) and Generalized Weighted Ensemble with Internally Tuned Hyperparameters (GEM–ITH).

The GEM was proposed to find the best combination of base learners by Perrone and Cooper (1992). It minimizes the total  mean square error (MSE) to optimize the weight of each base learner. The fomulation is shown as bellow:





Based on bias-variance decomposition (Hastie et al. 2005) the above definitions for bias and variance can be aggregated to the following:
$$ E[(f(x) -  \hat{f}(x))^2] = (Bias[\hat{f}(x)])^2 + Var[\hat{f}(x)] + Var(\epsilon) $$


Data Collection and Analysis Result
---

Conclusion
---

### Feature Work 
Tran et al. (2020) had proposed a framework of 

---

Reference
---

* Hastie, T., Tibshirani, R., Friedman, J., & Franklin, J. The elements of statistical learning: data mining, inference and prediction. The Mathematical Intelligencer, 27(2), 83-85, 2005.
* Lu, H. W., & Lee, C. Y. Kernel-Based Dynamic Ensemble Technique for Remaining Useful Life Prediction. IEEE Robotics and Automation Letters, 2021.
* Perrone, M. P., & Cooper, L. N. When networks disagree: Ensemble methods for hybrid neural networks: BROWN UNIV PROVIDENCE RI INST FOR BRAIN AND NEURAL SYSTEMS, 1992.
* Shahhosseini, M., Hu, G., & Pham, H. Optimizing ensemble weights and hyperparameters of machine learning models for regression problems. arXiv preprint arXiv:1908.05287, 2019.
* Tran, N., Schneider, J. G., Weber, I., & Qin, A. K. Hyper-parameter optimization in classification: To-do or not-to-do. Pattern Recognition, 103, 107245, 2020.
* Yang, L., & Shami, A. On hyperparameter optimization of machine learning algorithms: Theory and practice. Neurocomputing, 415, 295-316, 2020.



