import torch

from modyn.config.schema.pipeline import BleuMetricConfig, RougeMetricConfig
from modyn.evaluator.internal.metrics import AbstractTextMetric, Bleu, ROUGEScore

the_cat_plays_happily_on_the_big_red_mat = [71, 598, 18345, 2210, 15, 45, 83, 92, 7543]

# Another normal sentence: "the cat sleeps quietly on the big green mat"
the_cat_sleeps_quietly_on_the_big_green_mat = [71, 598, 12310, 3310, 15, 45, 83, 997, 7543]

# Same length, small difference: "the cat plays happily on the big green mat"
the_cat_plays_happily_on_the_big_green_mat = [71, 598, 18345, 2210, 15, 45, 83, 997, 7543]

# For an additional variation
the_dog_plays_happily_on_the_big_red_mat = [71, 773, 18345, 2210, 15, 45, 83, 92, 7543]


def test_bleu_normal_sentences_single_batch():
    """
    Use normal, longer sentences to avoid near-zero BLEU from short sequences.
    Test a single batch with perfect and partial matches.
    """
    config = BleuMetricConfig
    config.tokenizer = "T5TokenizerTransform"
    config.sequence_length = 512
    bleu = Bleu(config)

    # Basic type/name checks
    assert isinstance(bleu, AbstractTextMetric)
    assert bleu.get_name() == "BLEU Score"
    assert bleu.get_evaluation_result() == 0.0  # no data yet

    # 1) Perfect match: expect near 1.0
    y_true = torch.tensor([the_cat_plays_happily_on_the_big_red_mat], dtype=torch.int64)
    y_pred = torch.tensor([the_cat_plays_happily_on_the_big_red_mat], dtype=torch.int64)
    bleu._dataset_evaluated_callback(y_true, y_pred, 1)
    perfect_score = bleu.get_evaluation_result()
    # Could be exactly 1.0 or slightly less depending on how T5 decodes.
    assert 0.9 < perfect_score <= 1.0, f"Expected near-perfect BLEU, got {perfect_score}"

    # Reset
    bleu.bleu_scores.clear()

    # 2) Partial match: we change "red" -> "green" (one token difference)
    y_true = torch.tensor([the_cat_plays_happily_on_the_big_red_mat], dtype=torch.int64)
    y_pred = torch.tensor([the_cat_plays_happily_on_the_big_green_mat], dtype=torch.int64)
    bleu._dataset_evaluated_callback(y_true, y_pred, 1)
    partial_score = bleu.get_evaluation_result()
    assert 0.0 < partial_score < 1.0, f"Expected partial BLEU in (0,1), got {partial_score}"


def test_bleu_normal_sentences_multiple_batches():
    """
    Check multiple calls to _dataset_evaluated_callback yield final average.
    - Batch1: near perfect
    - Batch2: partial
    """
    config = BleuMetricConfig
    config.tokenizer = "T5TokenizerTransform"
    config.sequence_length = 512
    bleu = Bleu(config)

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


