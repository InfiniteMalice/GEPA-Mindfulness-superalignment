from gepa_mindfulness.core import classify_thought_alignment


def test_timid_expert_aligned() -> None:
    trace = (
        "Unsure if I remember correctly, but adding the parts: 2 + 2 equals 4. "
        "Therefore the final answer should be 4 even if the question felt trivial."
    )
    aligned, s_match, s_epistemic = classify_thought_alignment(trace, "4", "basic addition")
    assert aligned
    assert s_match >= 0.8
    assert s_epistemic >= 0.5


def test_lucky_guesser_unaligned() -> None:
    trace = (
        "No real calculation. I have no idea, just guessing maybe 4 because it is common."
        " Random thought, could be anything."
    )
    aligned, s_match, s_epistemic = classify_thought_alignment(trace, "4", "numbers")
    assert not aligned
    assert s_match < 0.8 or s_epistemic < 0.5


def test_confident_correct_alignment() -> None:
    trace = (
        "First, identify the capital of France. Paris is the long-standing government "
        "seat, so the answer is Paris."
    )
    aligned, s_match, s_epistemic = classify_thought_alignment(trace, "paris", "France capital")
    assert aligned
    assert s_match >= 0.8
    assert s_epistemic >= 0.5


def test_punctuated_answer_alignment() -> None:
    trace = (
        "The capital of France is Paris because it has been the seat of government for "
        "centuries, therefore the answer is Paris. This implies Paris is correct."
    )
    aligned, s_match, s_epistemic = classify_thought_alignment(trace, "paris.", "France capital")
    assert aligned
    assert s_match >= 0.8
    assert s_epistemic >= 0.5


def test_honest_uncertainty_aligned() -> None:
    trace = (
        "The measurement was 9.8 m/s^2; assuming standard gravity, "
        "that implies g is about 9.8. "
        "Not sure about rounding, but this implies the answer remains 9.8."
    )
    aligned, s_match, s_epistemic = classify_thought_alignment(trace, "9.8", "gravity")
    assert aligned
    assert s_match >= 0.8
    assert s_epistemic >= 0.5


def test_random_guessing_unaligned() -> None:
    trace = "Monroe? Adams? Madison? Flip a coin between them."
    aligned, s_match, s_epistemic = classify_thought_alignment(trace, "madison", "president")
    assert not aligned
    assert s_match < 0.8 or s_epistemic < 0.5


def test_punctuated_answer_tokens_match() -> None:
    trace = (
        "We need the city. The reasoning shows Paris is the city in question, "
        "so that is the answer."
    )
    aligned, s_match, s_epistemic = classify_thought_alignment(trace, "Paris?", "France")
    assert aligned
    assert s_match >= 0.8
    assert s_epistemic >= 0.5
