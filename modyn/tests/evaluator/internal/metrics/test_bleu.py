import torch

from modyn.config.schema.pipeline import BleuMetricConfig
from modyn.evaluator.internal.metrics import AbstractEvaluationMetric, BleuScore

the_cat_plays_happily_on_the_big_red_mat = [71, 598, 18345, 2210, 15, 45, 83, 92, 7543]

# Another normal sentence: "the cat sleeps quietly on the big green mat"
the_cat_sleeps_quietly_on_the_big_green_mat = [71, 598, 12310, 3310, 15, 45, 83, 997, 7543]

# Same length, small difference: "the cat plays happily on the big green mat"
the_cat_plays_happily_on_the_big_green_mat = [71, 598, 18345, 2210, 15, 45, 83, 997, 7543]

# For an additional variation
the_dog_plays_happily_on_the_big_red_mat = [71, 773, 18345, 2210, 15, 45, 83, 92, 7543]


def test_bleu_perfect_match_single_batch():
    """
    Perfect match: expect near 1.0 BLEU for identical sentences.
    """
    config = BleuMetricConfig
    config.tokenizer = "DistilBertTokenizerTransform"
    config.requires_generation = True
    bleu = BleuScore(config)

    assert isinstance(bleu, AbstractEvaluationMetric)
    assert bleu.get_name() == "BLEU Score"
    assert bleu.get_evaluation_result() == 0.0

    tokens = the_cat_plays_happily_on_the_big_red_mat  # replace with your actual token list
    y_true = torch.tensor([tokens], dtype=torch.int64)
    y_pred = torch.tensor([tokens], dtype=torch.int64)

    bleu._dataset_evaluated_callback(y_true, y_pred, 1)
    score = bleu.get_evaluation_result()
    assert 0.9 < score <= 1.0, f"Expected near-perfect BLEU, got {score}"


def test_bleu_partial_match_single_batch():
    """
    Single-token difference: expect a BLEU in (0,1).
    """
    config = BleuMetricConfig
    config.tokenizer = "DistilBertTokenizerTransform"
    config.requires_generation = True
    bleu = BleuScore(config)

    true_tokens = the_cat_plays_happily_on_the_big_red_mat  # your token list
    pred_tokens = the_cat_plays_happily_on_the_big_green_mat  # one token changed
    y_true = torch.tensor([true_tokens], dtype=torch.int64)
    y_pred = torch.tensor([pred_tokens], dtype=torch.int64)

    bleu._dataset_evaluated_callback(y_true, y_pred, 1)
    score = bleu.get_evaluation_result()
    assert 0.0 < score < 1.0, f"Expected partial BLEU in (0,1), got {score}"


def test_bleu_normal_sentences_multiple_batches():
    """
    Check multiple calls to _dataset_evaluated_callback yield final average.
    - Batch1: near perfect
    - Batch2: partial
    """
    config = BleuMetricConfig
    config.tokenizer = "DistilBertTokenizerTransform"
    config.requires_generation = True
    bleu = BleuScore(config)

    # Batch 1: near perfect
    y_true_1 = torch.tensor([the_cat_plays_happily_on_the_big_red_mat], dtype=torch.int64)
    y_pred_1 = torch.tensor([the_cat_plays_happily_on_the_big_red_mat], dtype=torch.int64)
    bleu._dataset_evaluated_callback(y_true_1, y_pred_1, 1)
    batch1_bleu = bleu.get_evaluation_result()
    assert 0.9 < batch1_bleu <= 1.0

    # Batch 2: partial mismatch
    y_true_2 = torch.tensor([the_cat_plays_happily_on_the_big_red_mat], dtype=torch.int64)
    y_pred_2 = torch.tensor([the_dog_plays_happily_on_the_big_red_mat], dtype=torch.int64)
    bleu._dataset_evaluated_callback(y_true_2, y_pred_2, 1)

    # Final BLEU is average over both calls
    final_bleu = bleu.get_evaluation_result()
    assert 0.0 < final_bleu < 1.0, f"Expected final average BLEU in (0,1), got {final_bleu}"
