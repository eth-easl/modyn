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
    config = BleuMetricConfig()
    bleu = Bleu(config, tokenizer="T5TokenizerTransform")

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
    config = BleuMetricConfig()
    bleu = Bleu(config, tokenizer="T5TokenizerTransform")

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


def test_rouge_normal_sentences_single_batch():
    """
    Test ROUGE with normal sentences using token ID input.
    - Perfect match => near 1.0
    - Partial match => score in (0,1)
    """
    config = RougeMetricConfig()
    rouge_metric = ROUGEScore(config, tokenizer="T5TokenizerTransform")

    assert isinstance(rouge_metric, AbstractTextMetric)
    assert rouge_metric.get_name() == "ROUGE Score"
    init_res = rouge_metric.get_evaluation_result()
    assert init_res == {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

    # Use token IDs for a full sentence
    y_true = torch.tensor([[71, 598, 18345, 2210, 15, 45, 83, 92, 7543]])  # "the cat plays happily on the big red mat"
    y_pred = torch.tensor([[71, 598, 18345, 2210, 15, 45, 83, 92, 7543]])
    rouge_metric._dataset_evaluated_callback(y_true, y_pred, 1)
    res_perfect = rouge_metric.get_evaluation_result()
    for val in res_perfect.values():
        assert 0.9 < val <= 1.0, f"Expected near-perfect ROUGE, got {val}"

    # Reset
    rouge_metric.scores = {"rouge-1": [], "rouge-2": [], "rouge-l": []}

    # Partial match — one token changed
    y_true2 = torch.tensor([[71, 598, 18345, 2210, 15, 45, 83, 92, 7543]])
    y_pred2 = torch.tensor([[71, 598, 18345, 2210, 15, 45, 83, 997, 7543]])  # red → green
    rouge_metric._dataset_evaluated_callback(y_true2, y_pred2, 1)
    res_partial = rouge_metric.get_evaluation_result()
    for val in res_partial.values():
        assert 0.0 < val < 1.0, f"Expected partial ROUGE in (0,1), got {val}"


def test_rouge_normal_sentences_multiple_batches():
    """
    Test ROUGE handles multiple batches using token ID input.
    Final score must be average across both.
    """
    config = RougeMetricConfig()
    rouge_metric = ROUGEScore(config, tokenizer="T5TokenizerTransform")

    # Batch 1: perfect match
    y_true_1 = torch.tensor([[71, 598, 18345, 2210, 15, 45, 83, 92, 7543]])
    y_pred_1 = torch.tensor([[71, 598, 18345, 2210, 15, 45, 83, 92, 7543]])
    rouge_metric._dataset_evaluated_callback(y_true_1, y_pred_1, 1)
    batch1 = rouge_metric.get_evaluation_result()
    for v in batch1.values():
        assert 0.9 < v <= 1.0

    # Batch 2: partial match (quietly vs happily)
    y_true_2 = torch.tensor([[71, 598, 18345, 2210, 15, 45, 83, 92, 7543]])
    y_pred_2 = torch.tensor([[71, 598, 12310, 3310, 15, 45, 83, 92, 7543]])  # cat sleeps quietly...
    rouge_metric._dataset_evaluated_callback(y_true_2, y_pred_2, 1)

    final_res = rouge_metric.get_evaluation_result()
    for v in final_res.values():
        assert 0.0 < v < 1.0, f"Expected final average ROUGE in (0,1), got {v}"
