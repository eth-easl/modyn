# In the "forecast" mode, the decision is made based on the current performance, future performance estimates,
# the cumulated avoidable misclassifications. We use a sort of break even analysis to estimate the time point
# in the future where cumulated regret equals to the retraining cost. If this time point is within the current
# update point and the next update point, we trigger.
