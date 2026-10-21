import json
import random
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from synthia.retrieval.wordpiece import (
    END,
    MAX_WORD_CHARS,
    START,
    UNKNOWN,
    TokenizerError,
    WordPiece,
    normalized,
    words,
)

SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
# Characters every rule has an opinion on: controls, odd whitespace, accents,
# Greek sigma, CJK, punctuation of every kind, emoji and bare combining marks.
POOLS = (
    "abcdefghijklmnopqrstuvwxyz ABCXYZ 0123456789",
    "     \t\n\r",
    "!\"#$%&'()*+,-./:;<=>?@\\^_`{|}~",
    "\u00e9\u00e8\u00ea\u00fc\u00f1\u00e7\u00c5\u00d8\u00df\u0130\u0131",
    "\u03a3\u03c3\u03c2\u0391\u0392\u0393",
    "\u4e2d\u6587\u5b57\u3042\u30a2\ud55c\uae00",
    "\u0000\u0001\u001c\u001f\u007f\u0085\u00a0\u200b\u200d\u2028\u3000\ufeff\ufffd\ue000",
    "\u2014\u2013\u201c\u201d\u00bf\u00a1\u3002\u300c",
    "\U0001f600\U0001f44d\u2764\ufe0f",
    "\u0301\u0308\u0327",
)


def made_up_text(rng: random.Random) -> str:
    pools = rng.sample(POOLS, rng.randint(1, 4))
    return "".join(rng.choice(rng.choice(pools)) for _ in range(rng.randint(1, 60)))


def bert_spec(vocabulary: dict[str, int], max_tokens: int) -> str:
    """A tokenizer.json with BERT's rules, in the form embedding models ship it."""

    def special(token: str) -> dict[str, object]:
        return {"id": token, "ids": [vocabulary[token]], "tokens": [token]}

    wrapped = [
        {"SpecialToken": {"id": START, "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": END, "type_id": 0}},
    ]
    spec: dict[str, object] = {
        "version": "1.0",
        "truncation": {
            "direction": "Right",
            "max_length": max_tokens,
            "strategy": "LongestFirst",
            "stride": 0,
        },
        "padding": None,
        "added_tokens": [],
        "normalizer": {
            "type": "BertNormalizer",
            "clean_text": True,
            "handle_chinese_chars": True,
            "strip_accents": None,
            "lowercase": True,
        },
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": wrapped,
            "pair": [
                *wrapped,
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": END, "type_id": 1}},
            ],
            "special_tokens": {START: special(START), END: special(END)},
        },
        "decoder": None,
        "model": {
            "type": "WordPiece",
            "unk_token": UNKNOWN,
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": MAX_WORD_CHARS,
            "vocab": vocabulary,
        },
    }
    return json.dumps(spec)


@pytest.fixture(scope="module")
def vocabulary() -> dict[str, int]:
    """Whole words, single characters and ## pieces seen in the made-up text."""
    rng = random.Random(1)
    pieces: set[str] = set()
    for _ in range(2000):
        for word in words(normalized(made_up_text(rng))):
            pieces.update((word, word[:3], *word))
            pieces.update(f"##{word[i:]}" for i in range(1, len(word)))
            pieces.update(f"##{c}" for c in word[1:])
    return {t: i for i, t in enumerate((*SPECIALS, *sorted(pieces - set(SPECIALS))))}


@pytest.fixture(scope="module")
def tokenizer_file(
    vocabulary: dict[str, int], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    path.write_text(bert_spec(vocabulary, 64), encoding="utf-8")
    return path


def test_ids_are_the_ones_the_tokenizers_library_makes(tokenizer_file: Path) -> None:
    ours = WordPiece.from_file(tokenizer_file)
    theirs = Tokenizer.from_file(str(tokenizer_file))
    rng = random.Random(2)
    texts = [made_up_text(rng) for _ in range(5000)]

    assert [ours.ids(t, 64) for t in texts] == [theirs.encode(t).ids for t in texts]


def made_up_pairs(seed: int, count: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    return [
        (made_up_text(rng) * rng.randint(1, 3), made_up_text(rng) * rng.randint(1, 3))
        for _ in range(count)
    ]


def lengths_decide(ours: WordPiece, a: str, b: str, max_tokens: int) -> bool:
    """Whether the pair's cut follows from the two lengths alone.

    Only when both texts are cut to half an odd room and neither is under
    ``max_tokens`` pieces does tokenizers 0.23 hand the odd piece to one side
    or the other by how the pieces group into words.
    """
    first, second, room = ours.count(a), ours.count(b), max_tokens - 3
    halved = 2 * min(first, second) > room and first + second > room
    return not halved or room % 2 == 0 or min(first, second) < max_tokens


@pytest.mark.parametrize("max_tokens", [64, 16, 9])
def test_pairs_are_the_ones_the_tokenizers_library_makes(
    vocabulary: dict[str, int], max_tokens: int
) -> None:
    ours = WordPiece(vocabulary)
    theirs = Tokenizer.from_str(bert_spec(vocabulary, max_tokens))
    pairs = [
        (a, b)
        for a, b in made_up_pairs(max_tokens, 3000)
        if lengths_decide(ours, a, b, max_tokens)
    ]

    ours_encoded = [ours.pair(a, b, max_tokens) for a, b in pairs]
    theirs_encoded = [theirs.encode(a, b) for a, b in pairs]

    assert len(pairs) > 1500
    assert [(p.ids, p.types) for p in ours_encoded] == [
        (tuple(e.ids), tuple(e.type_ids)) for e in theirs_encoded
    ]


@pytest.mark.parametrize("max_tokens", [16])
def test_an_odd_piece_the_lengths_do_not_decide_moves_at_most_one_piece(
    vocabulary: dict[str, int], max_tokens: int
) -> None:
    ours = WordPiece(vocabulary)
    theirs = Tokenizer.from_str(bert_spec(vocabulary, max_tokens))
    pairs = [
        (a, b)
        for a, b in made_up_pairs(max_tokens, 3000)
        if not lengths_decide(ours, a, b, max_tokens)
    ]

    kept = [
        (
            ours.pair(a, b, max_tokens).types.count(0),
            theirs.encode(a, b).type_ids.count(0),
        )
        for a, b in pairs
    ]

    assert len(pairs) > 100
    assert all(abs(mine - library) <= 1 for mine, library in kept)
    assert all(len(ours.pair(a, b, max_tokens).ids) == max_tokens for a, b in pairs)


LETTERS = {t: i for i, t in enumerate((*SPECIALS, "a", "b"))}


def test_a_pair_is_both_texts_between_the_special_tokens() -> None:
    pair = WordPiece(LETTERS).pair("a", "b b", 64)
    a, b, start, end = LETTERS["a"], LETTERS["b"], LETTERS[START], LETTERS[END]

    assert pair.ids == (start, a, end, b, b, end)
    assert pair.types == (0, 0, 0, 1, 1, 1)


@pytest.mark.parametrize(
    ("first", "second", "room", "kept"),
    [
        (2, 9, 7, (2, 5)),
        (9, 2, 7, (5, 2)),
        (5, 5, 7, (3, 4)),
        (6, 5, 7, (4, 3)),
        (11, 10, 7, (3, 4)),
        (4, 9, 6, (3, 3)),
    ],
)
def test_a_pair_cuts_the_longer_text_first(
    first: int, second: int, room: int, kept: tuple[int, int]
) -> None:
    pair = WordPiece(LETTERS).pair(
        " ".join("a" * first), " ".join("b" * second), room + 3
    )

    assert (pair.ids.count(LETTERS["a"]), pair.ids.count(LETTERS["b"])) == kept


def test_a_pair_needs_room_inside_its_three_special_tokens() -> None:
    with pytest.raises(ValueError, match=r"^max_tokens=3 leaves no room"):
        WordPiece(LETTERS).pair("a", "b", 3)


def test_count_is_the_number_of_pieces_the_tokenizers_library_makes_uncut(
    tokenizer_file: Path,
) -> None:
    ours = WordPiece.from_file(tokenizer_file)
    theirs = Tokenizer.from_file(str(tokenizer_file))
    theirs.no_truncation()
    rng = random.Random(3)
    texts = [made_up_text(rng) * 4 for _ in range(1000)]

    assert [ours.count(t) for t in texts] == [
        len(theirs.encode(t).ids) - 2 for t in texts
    ]
    assert max(ours.count(t) for t in texts) > 64


def test_count_matches_ids_inside_the_start_and_end_tokens() -> None:
    tokens = WordPiece(
        {t: i for i, t in enumerate((*SPECIALS, "un", "##aff", "##able", "x"))}
    )

    assert tokens.count("Unaffable x\nx") == 5
    assert tokens.count("unaffable x x") == len(tokens.ids("unaffable x x", 64)) - 2
    assert tokens.count(" \n\t") == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Caf\u00e9 \u00dcBER", "cafe uber"),
        ("a\u001cb\u0085c", "abc"),
        ("a\tb\u00a0c", "a b c"),
        ("\u039f\u0394\u039f\u03a3", "\u03bf\u03b4\u03bf\u03c3"),
        ("\u4e2d\u6587", " \u4e2d  \u6587 "),
    ],
    ids=["accents", "controls", "whitespace", "final sigma", "cjk"],
)
def test_text_is_normalised_as_bert_does(text: str, expected: str) -> None:
    assert normalized(text) == expected


def test_words_split_on_spaces_and_around_each_punctuation_mark() -> None:
    assert words("don't stop\u2014now!!") == [
        "don",
        "'",
        "t",
        "stop",
        "\u2014",
        "now",
        "!",
        "!",
    ]


def test_a_word_splits_into_the_longest_known_pieces() -> None:
    tokens = WordPiece(
        {t: i for i, t in enumerate((*SPECIALS, "un", "##aff", "##able", "##a", "x"))}
    )

    assert tokens.pieces("unaffable") == [5, 6, 7]
    assert tokens.pieces("unxx") == [1]
    assert tokens.pieces("x" * (MAX_WORD_CHARS + 1)) == [1]
    assert tokens.ids("Unaffable x x x", max_tokens=5) == [2, 5, 6, 7, 3]


def test_text_spelling_a_special_token_stays_text() -> None:
    tokens = WordPiece({t: i for i, t in enumerate((*SPECIALS, "sep", "[", "]"))})
    assert tokens.ids("[SEP]", 16) == [2, 6, 5, 7, 3]


def test_a_vocabulary_without_the_special_tokens_is_refused() -> None:
    with pytest.raises(TokenizerError, match=r"\[UNK\]"):
        WordPiece({"[CLS]": 0, "[SEP]": 1})


@pytest.mark.parametrize(
    ("part", "key", "value", "message"),
    [
        ("normalizer", "lowercase", False, "lowercase"),
        ("normalizer", "strip_accents", False, "accents"),
        ("pre_tokenizer", "type", "Whitespace", "pre_tokenizer"),
        ("model", "vocab", None, "not a WordPiece"),
    ],
    ids=["cased", "accents kept", "other split", "no vocabulary"],
)
def test_a_tokenizer_with_other_rules_is_refused(  # noqa: PLR0913
    tokenizer_file: Path,
    tmp_path: Path,
    *,
    part: str,
    key: str,
    value: object,
    message: str,
) -> None:
    spec: dict[str, dict[str, object]] = json.loads(
        tokenizer_file.read_text(encoding="utf-8")
    )
    if value is None:
        del spec[part][key]
    else:
        spec[part][key] = value
    other = tmp_path / "tokenizer.json"
    other.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(TokenizerError, match=message):
        WordPiece.from_file(other)
