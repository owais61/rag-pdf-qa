import json
import os

from eval.evaluate import print_table, run_eval
from tests.conftest import ROOT


def test_questions_file_is_valid():
    with open(os.path.join(ROOT, "eval", "questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    assert len(questions) >= 10
    assert all({"question", "answer_contains"} <= set(q) for q in questions)


def test_every_expected_phrase_exists_in_the_sample_pdf(sample_pdf):
    """If a phrase is missing from the PDF, the eval could never score a hit."""
    from eval.evaluate import _norm
    from src.ingestion import load_pdf

    text = _norm(" ".join(d.page_content for d in load_pdf(sample_pdf)))
    with open(os.path.join(ROOT, "eval", "questions.json"), encoding="utf-8") as f:
        for q in json.load(f):
            assert _norm(q["answer_contains"]) in text, q


def test_run_eval_produces_scores_for_every_mode(embedder, sample_pdf, capsys):
    with open(os.path.join(ROOT, "eval", "questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    results = run_eval(embedder, sample_pdf, questions, ks=(1, 3, 5))

    assert set(results) == {"dense", "bm25", "hybrid"}
    for row in results.values():
        assert row["n"] == len(questions)
        assert 0 <= row["hit@1"] <= row["hit@3"] <= row["hit@5"] <= 1
        assert 0 <= row["mrr"] <= 1
    assert results["hybrid"]["hit@5"] >= 0.5  # the pipeline finds most answers

    print_table(results, [1, 3, 5])
    assert "hybrid" in capsys.readouterr().out
