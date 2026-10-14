from synthia.retrieval.fts import match_all_expression, match_expression


def test_each_word_is_quoted_and_any_may_match() -> None:
    assert match_expression('tell me "about" Asha!') == (
        '"tell" OR "me" OR "about" OR "Asha"'
    )
    assert match_expression("?!") == ""


def test_each_word_is_quoted_and_all_must_match() -> None:
    assert match_all_expression("Asha's cat") == '"Asha" AND "s" AND "cat"'
