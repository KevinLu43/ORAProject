# ORA Project


## Optimizing Ensemble Weights and Hyper Parameters of Machine Learning Models
---

### Background and Motivation
Nowadays, machine learning has become popular in every area. There always are novel methods been proposed, however, there usually are some parameters or hyper parameters need to be set in order to construct the model. The suitable hyper parameters for the model may affect the performance of the model.

On the other hand, aggregating multiple models which were called ensemble method also been widely used in recent years.Basic ensemble method combines several regression base learners by averaging their estimates. Another ensemble method uses the linear combination of the regression base learners. The weights are assigned to each of the base learners according to their performance on the validation set so called weighted majority approach is the other idea to integrate all predictions.

Therefore, how to assign the proper ensemble wights and set hyper parameters of machine learning models is the topic, Shahhosseini and Phamthere (2019) proposed a framework that combine the model hyperparameter tuning and the model weights aggregation for optimal ensemble design in one coherent process, we follow the framework and try to implement the method in the remaining useful life prediction.
