
"""Unit tests for the evaluation code itself (no LLM, no API key, no MLflow needed).

If your scoring code is wrong, every eval number is meaningless, so we test it too.
"""
from evals.run_eval import evaluate, parse_judge_reply, score_keywords


def test_keyword_match_all_words_required():
    assert score_keywords("Marwa founded it in 1873", ["Marwa", "1873"], False)
    assert not score_keywords("Marwa founded it", ["Marwa", "1873"], False)


def test_keyword_match_is_case_insensitive():
    assert score_keywords("the capital is QUILLON", ["Quillon"], False)


def test_refusal_detected():
    assert score_keywords("Not in the provided documents", [], True)
    assert not score_keywords("Paris is the capital", [], True)


def test_judge_reply_parsing():
    assert parse_judge_reply('{"faithfulness": 4, "relevance": 5}') == (4, 5)
    assert parse_judge_reply('Sure! {"faithfulness":5,"relevance":3} done') == (5, 3)
    assert parse_judge_reply("no idea") is None
    assert parse_judge_reply('{"faithfulness": 9, "relevance": 5}') is None


def test_evaluate_summary_with_stub_pipeline():
    dataset = [
        {"id": "a", "question": "capital?", "expected": ["Quillon"], "should_refuse": False},
        {"id": "b", "question": "currency?", "expected": ["Zor"], "should_refuse": False},
        {"id": "c", "question": "world cup?", "expected": [], "should_refuse": True},
    ]
    answers = {"capital?": "Quillon", "currency?": "Dollar", "world cup?": "Not in the provided documents"}
    rows, summary = evaluate(dataset, lambda q: (answers[q], ["ctx"]),
                             judge_fn=lambda q, c, a: '{"faithfulness": 4, "relevance": 5}')
    assert summary["answer_accuracy"] == 0.5      # 1 of 2 answerable correct
    assert summary["refusal_accuracy"] == 1.0
    assert summary["avg_faithfulness"] == 4
    assert summary["error_count"] == 0


def test_evaluate_survives_pipeline_errors():
    def boom(q):
        raise RuntimeError("rate limit")
    rows, summary = evaluate([{"id": "a", "question": "x", "expected": ["y"], "should_refuse": False}], boom)
    assert summary["error_count"] == 1
    assert rows[0]["correct"] is False

def test_keyword_match_ignores_odd_unicode_spaces():
    assert score_keywords("2\u202fOctober", ["2 October"], False)
    assert score_keywords("2\u00a0October", ["2 October"], False)    