
from modyn.supervisor.internal.triggers.drift.classifier_models.ybnet_classifier import YearbookNetDriftDetector


alibi_classifier_models = {
    "ybnet": YearbookNetDriftDetector(3),
}
