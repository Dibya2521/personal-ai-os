import json
import random
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from synthia.memory.wordpiece import (
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
            "pair": wrapped,
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
