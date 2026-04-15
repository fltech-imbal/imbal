# Using `imbal.metric` In Model Classification

In model classification, metrics can be used to track model performance.
Keras includes a variety of metric object, and `imbal` provides several
additional metrics for use, a full list of which can be found on 
[this page](../../../metrics/metrics.md). This tutorial will explain some
of the ways to use metrics, along with code examples.

Throughout this tutorial, we will make use of the following metrics:
- [keras.metrics.F1Score](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/F1Score)
- [imbal.metrics.HeikdeSkillScore](../../../metrics/heikde_skill_score.md)
- [imbal.metrics.TrueSkillStatistic](../../../metrics/true_skill_statistic.md)
- [keras.metrics.AUC](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC)

