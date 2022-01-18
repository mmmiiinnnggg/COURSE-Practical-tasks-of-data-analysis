https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists

https://www.kaggle.com/lavanyashukla01/battle-of-the-boosting-algos-lgb-xgb-catboost/

[大战三回合：XGBoost、LightGBM和Catboost一决高低 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/72686522?utm_source=wechat_session&utm_medium=social&utm_oi=915021263796322304&utm_campaign=shareopn)



[(31条消息) 【机器学习】Optuna机器学习模型调参(LightGBM、XGBoost)_ccql's Blog-CSDN博客_optuna](https://blog.csdn.net/qq_43510916/article/details/113794486)



- **CatBoost**



（1）CatBoost 提供了比 XGBoost 更高的准确性和和更短的训练时间；

（2）支持即用的分类特征，因此我们不需要对分类特征进行预处理（例如，通过 LabelEncoding 或 OneHotEncoding）。事实上，CatBoost 的文档明确地说明不要在预处理期间使用热编码，因为“这会影响训练速度和最终的效果”；

（3）通过执行有序地增强操作，可以更好地处理过度拟合，尤其体现在小数据集上；

（4）支持即用的 GPU 训练（只需设置参数task_type =“GPU”）；

（5）可以处理缺失的值；

- **LightGBM**



（1）LightGBM 也能提供比 XGBoost 更高的准确性和更短的训练时间；

（2）支持并行的树增强操作，即使在大型数据集上（相比于 XGBoost）也能提供更快的训练速度；

（3）使用 histogram-esquealgorithm，将连续的特征转化为离散的特征，从而实现了极快的训练速度和较低的内存使用率；

（4）通过使用垂直拆分（leaf-wise split）而不是水平拆分（level-wise split）来获得极高的准确性，这会导致非常快速的聚合现象，并在非常复杂的树结构中能捕获训练数据的底层模式。可以通过使用 num_leaves 和 max_depth 这两个超参数来控制过度拟合；

- **XGBoost**



（1）支持并行的树增强操作；

（2）使用规则化来遏制过度拟合；

（3）支持用户自定义的评估指标；

（4）处理缺失的值；

（5）XGBoost 比传统的梯度增强方法（如 AdaBoost）要快得多；



下面列出的是模型中一些重要的参数，以帮助大家更好学习与使用这些算法！

- **Catboost**

- - n_estimators：表示用于创建树的最大数量；
  - learning_rate：表示学习率，用于减少梯度的级别；
  - eval_metric：表示用于过度拟合检测和最佳模型选择的度量标准；
  - depth：表示树的深度；
  - subsample：表示数据行的采样率，不能在贝叶斯增强类型设置中使用；
  - l2_leaf_reg：表示成本函数的L2规则化项的系数；
  - random_strength：表示在选择树结构时用于对拆分评分的随机量，使用此参数可以避免模型过度拟合；
  - min_data_in_leaf：表示在一个叶子中训练样本的最小数量。CatBoost不会在样本总数小于指定值的叶子中搜索新的拆分；
  - colsample_bylevel, colsample_bytree, colsample_bynode — 分别表示各个层、各棵树、各个节点的列采样率；
  - task_type：表示选择“GPU”或“CPU”。如果数据集足够大（从数万个对象开始），那么在GPU上的训练与在CPU上的训练相比速度会有显著的提升，数据集越大，加速就越明显；
  - boosting_type：表示在默认情况下，小数据集的增强类型值设置为“Ordered”。这可以防止过度拟合，但在计算方面的成本会很高。可以尝试将此参数的值设置为“Plain”，来提高训练速度；
  - rsm：对于那些具有几百个特性的数据集，rsm参数加快了训练的速度，通常对训练的质量不会有影响。另外，不建议为只有少量（10-20）特征的数据集更改rsm参数的默认值；
  - border_count：此参数定义了每个特征的分割数。默认情况下，如果在CPU上执行训练，它的值设置为254，如果在GPU上执行训练，则设置为128；

- **LightGBM**

- - num_leaves：表示一棵树中最大的叶子数量。在LightGBM中，必须将num_leaves的值设置为小于2^（max_depth），以防止过度拟合。而更高的值会得到更高的准确度，但这也可能会造成过度拟合；

  - max_depth：表示树的最大深度，这个参数有助于防止过度拟合；

  - min_data_in_leaf：表示每个叶子中的最小数据量。设置一个过小的值可能会导致过度拟合；

  - eval_metric：表示用于过度拟合检测和最佳模型选择的度量标准；

  - learning_rate：表示学习率，用于降低梯度的级别；

  - n_estimators：表示可以创建树的最大数量；

  - colsample_bylevel, colsample_bytree, colsample_bynode — 分别表示各个层、各棵树、各个节点的列采样率；

  - boosting_type — 该参数可选择以下的值:

  - - ‘gbdt’,表示传统的梯度增强决策树；
    - ‘dart’,缺失则符合多重累计回归树（Multiple Additive Regression Trees）；
    - ‘goss’,表示基于梯度的单侧抽样（Gradient-based One-Side Sampling）；
    - ‘rf’,表示随机森林（Random Forest）；

  - feature_fraction：表示每次迭代所使用的特征分数（即所占百分比，用小数表示）。将此值设置得较低，来提高训练速度；

  - min_split_again：表示当在树的叶节点上进行进一步的分区时，所需最小损失值的减少量；

  - n_jobs：表示并行的线程数量，如果设为-1则可以使用所有的可用线程；

  - bagging_fraction：表示每次迭代所使用的数据分数（即所占百分比，用小数表示）。将此值设置得较低，以提高训练速度；

  - application ：default（默认值）=regression, type（类型值）=enum, options（可选值）=

  - - regression : 表示执行回归任务；
    - binary : 表示二进制分类；
    - multiclass:表示多个类的类别；
    - lambdarank : 表示lambdarank 应用；

  - max_bin：表示用于存放特征值的最大容器（bin）数。有助于防止过度拟合；

  - num_iterations：表示增强要执行的迭代的迭代；

