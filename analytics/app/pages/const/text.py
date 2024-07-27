COMPOSITE_MODEL_TEXT = """
    ## Composite model variant

    The composite model is the pipeline model that is made up by patching together the individual models
    of the pipeline. We support two variants of the composite model:
    - `currently_active_model`: For a certain point in time we make a fixed model the `pipeline` pipeline model,
        that shows up in the composite model, iff it is the most recent model which was trained on an interval
        that is strictly before the point of evaluation.
    - `currently_trained_model`: For a fixed point in time this is the model that was trained after the
        `currently_active_model`. So it is the model which training / training sample collection is still
        ongoing during the point of evaluation.
"""
