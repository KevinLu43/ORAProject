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
- Tran, N., Schneider, J. G., Weber, I., & Qin, A. K. Hyper-parameter optimization in classification: To-do or not-to-do. Pattern Recognition, 103, 107245, 2020.
- Yang, L., & Shami, A. On hyperparameter optimization of machine learning algorithms: Theory and practice. Neurocomputing, 415, 295-316, 2020.
- Shahhosseini, M., Hu, G., & Pham, H. Optimizing ensemble weights and hyperparameters of machine learning models for regression problems. arXiv preprint arXiv:1908.05287, 2019.
- Lu, H. W., & Lee, C. Y. Kernel-Based Dynamic Ensemble Technique for Remaining Useful Life Prediction. IEEE Robotics and Automation Letters, 2021.

