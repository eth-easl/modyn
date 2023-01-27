# TODO(MaxiBoether): implement.
# Idea: With reset, we always spit out the data since last trigger.
# If we have a limit, we choose a random subset from the data.
# This can be used to either continously finetune or retrain from scratch on only new data.
# Without reset, we always spit out the entire new data.
# If we do not have a limit, this can be used to retrain from scratch on all data,
# finetuning does not really make sense here, unless you want to revisit all the time.
# If we have a limit and no reset, we have multiple options and need to decide on one: "last X samples",
# sample uniform at random from all data, sample from all data but prioritize newer data points in some way
