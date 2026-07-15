# SPDX-License-Identifier: Apache-2.0
from service.dedup import cosine, normalize_question, problem_id_of


def test_normalize_collapses_whitespace_and_nfc():
    assert normalize_question("  foo   bar\n\tbaz  ") == "foo bar baz"
    # NFC: composed form equals decomposed form after normalization
    assert normalize_question("é") == normalize_question("é")


def test_normalize_preserves_case():
    assert normalize_question("Run Bash") == "Run Bash"
    assert normalize_question("Bash") != "bash"


def test_problem_id_stable_for_same_question():
    assert problem_id_of("hello world") == problem_id_of("hello world")


def test_problem_id_case_sensitive():
    assert problem_id_of("Bash") != problem_id_of("bash")


def test_problem_id_whitespace_normalized_to_same():
    assert problem_id_of("  hello   world  ") == problem_id_of("hello world")
    assert problem_id_of("hello\nworld") == problem_id_of("hello world")


def test_problem_id_length():
    assert len(problem_id_of("anything")) == 16


def test_cosine_orthogonal_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_same_direction_is_one():
    assert abs(cosine([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) - 1.0) < 1e-9


def test_cosine_zero_vector_is_zero():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine([1.0, 1.0], [0.0, 0.0]) == 0.0
