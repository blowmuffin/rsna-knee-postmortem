Submissions are evaluated by a weighted multilabel [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probabilities and the observed targets.

For each of the fourteen target labels, we compute the [AUC ROC score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html). The score for `Aneurysm Present` is assigned a weight of 13, while all the other scores are assigned a weight of 1. The final score is the average of the fourteen AUC ROC scores with these weights.

In other words, the final score is the simple average of the `Aneurysm Present` score and the average of the other thirteen scores:

$$
\text{Final Score} = \frac{1}{2}\left(AUC_{AP} + \frac{1}{13}\sum_{i=0}^{12} AUC_i\right)
$$

You may find the metric code here: [Mean Weighted Columnwise AUCROC](https://www.kaggle.com/code/metric/mean-weighted-columnwise-aucroc).

## Submission File ##

You must submit to this competition using the provided evaluation API. See [this example notebook](https://www.kaggle.com/code/ryanholbrook/rsna-aneurysm-detection-demo-submission) for more details.