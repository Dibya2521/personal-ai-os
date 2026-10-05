import random
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from synthia.memory.bm25 import Bm25Index, Hit, terms
from synthia.memory.porter import stem
from synthia.memory.store import FTS_TOKENIZER

REPO = Path(__file__).parents[2]
LETTERS = "aeiouybcdfghjklmnpqrstvwxz"  # pragma: allowlist secret
# Suffixes every step of the stemmer looks for, and pieces of them.
SUFFIXES = (
    *("", "s", "es", "ies", "sses", "ss", "ed", "eed", "ing", "y", "e", "ll"),
    *("at", "bl", "iz", "ational", "tional", "enci", "anci", "izer", "logi", "bli"),
    *("alli", "entli", "eli", "ousli", "ization", "ation", "ator", "alism"),
    *("iveness", "fulness", "ousness", "aliti", "iviti", "biliti", "icate"),
    *("ative", "alize", "iciti", "ical", "ful", "ness", "al", "ance", "ence"),
    *("er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "sion", "tion"),
    *("ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"),
)
# Letters outside a to z, two of them ending in a repeated UTF-8 byte.
OTHERS = ("\N{CYRILLIC SMALL LETTER ZHE}", "俿", "一", "é", "ß", "7")


def fts5_terms(texts: list[str]) -> list[list[bytes]]:
    """Return the terms SQLite's FTS5 makes of each text, in order."""
    connection = sqlite3.connect(":memory:")
    connection.text_factory = bytes
    connection.execute(
        f"CREATE VIRTUAL TABLE t USING fts5(a, tokenize='{FTS_TOKENIZER}')"
    )
    connection.execute("CREATE VIRTUAL TABLE v USING fts5vocab(t, 'instance')")
    connection.executemany(
        "INSERT INTO t(rowid, a) VALUES (?, ?)", enumerate(texts, start=1)
    )
    found: list[list[bytes]] = [[] for _ in texts]
    for term, row in connection.execute("SELECT term, doc FROM v ORDER BY doc, offset"):
        found[int(row) - 1].append(term)
    connection.close()
    return found


def made_up_word(rng: random.Random) -> str:
    start = "".join(rng.choice(LETTERS) for _ in range(rng.randint(0, 6)))
    if rng.random() < 0.1:
        start += rng.choice(OTHERS) * rng.randint(1, 2)
    return start + rng.choice(SUFFIXES) + rng.choice(SUFFIXES[:12])


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("cats", "cat"),
        ("its", "it"),
        ("as", "as"),
        ("sky", "sky"),
        ("skies", "ski"),
        ("happy", "happi"),
        ("running", "run"),
        ("hopping", "hop"),
        ("filing", "file"),
        ("generalization", "gener"),
        ("a" * 70 + "ing", "a" * 70 + "ing"),
    ],
)
def test_words_stem_as_porter_wrote(word: str, expected: str) -> None:
    assert stem(word.encode()) == expected.encode()


def test_a_word_is_stemmed_as_its_utf8_bytes() -> None:
    zhe = "\N{CYRILLIC SMALL LETTER ZHE}"
    # Two equal letters whose last two bytes differ: nothing is dropped.
    assert stem(f"a{zhe}{zhe}ing".encode()) == f"a{zhe}{zhe}".encode()
    # One letter whose last two bytes are equal: one byte is dropped.
    assert stem("a俿ing".encode()) == b"a\xe4\xbf"


def test_text_becomes_lower_case_terms_without_accents() -> None:
    assert terms("Don't RUN, Café_bar 42!") == [
        b"don",
        b"t",
        b"run",
        b"cafe",
        b"bar",
        b"42",
    ]


def test_terms_are_the_ones_fts5_makes() -> None:
    rng = random.Random(7)
    words = [made_up_word(rng) for _ in range(20_000)]
    paths = [
        *sorted(REPO.glob("docs/**/*.md")),
        REPO / "README.md",
        REPO / "CHANGELOG.md",
    ]
    documents = [path.read_text(encoding="utf-8") for path in paths]
    texts = [w for w in words if w] + documents

    assert [terms(text) for text in texts] == fts5_terms(texts)


class Corpus:
    """The same documents in the index and in an FTS5 table like the store's."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.index = Bm25Index()
        self.table = sqlite3.connect(":memory:")
        self.table.execute(
            "CREATE VIRTUAL TABLE t USING fts5"
            f"(question, answer, calls, tokenize='{FTS_TOKENIZER}')"
        )
        words = {made_up_word(rng) for _ in range(800)}
        self.words = sorted(w for w in words if w.isascii() and w)
        # Word frequencies fall off as in real text: a few words are everywhere.
        self.weights = [1 / rank for rank in range(1, len(self.words) + 1)]

    def text(self, count: int) -> str:
        return " ".join(self.rng.choices(self.words, self.weights, k=count))

    def put(self, document: int) -> None:
        question, answer = self.text(self.rng.randint(1, 30)), self.text(100)
        calls = self.text(3) if self.rng.random() < 0.3 else ""
        self.table.execute("DELETE FROM t WHERE rowid = ?", (document,))
        self.table.execute(
            "INSERT INTO t(rowid, question, answer, calls) VALUES (?, ?, ?, ?)",
            (document, question, answer, calls),
        )
        self.index.add(document, f"{question} {answer} {calls}")

    def remove(self, document: int) -> None:
        self.table.execute("DELETE FROM t WHERE rowid = ?", (document,))
        assert self.index.remove(document)

    def close(self) -> None:
        self.table.close()

    def assert_same_scores(self, queries: int) -> None:
        for _ in range(queries):
            words = self.rng.sample(self.words, self.rng.randint(1, 4))
            words += words[:1] if self.rng.random() < 0.2 else ["qqqzzz"]
            match = " OR ".join(f'"{word}"' for word in words)
            fts5 = dict(
                self.table.execute(
                    "SELECT rowid, -bm25(t) FROM t WHERE t MATCH ?", (match,)
                )
            )
            assert self.index.scores(" ".join(words)) == pytest.approx(fts5, rel=1e-12)


def test_scores_are_fts5_scores_through_adds_replaces_and_removals() -> None:
    rng = random.Random(11)
    with closing(Corpus(rng)) as corpus:
        for document in range(1, 501):
            corpus.put(document)
        corpus.assert_same_scores(100)

        for document in rng.sample(range(1, 501), 100):
            corpus.remove(document)
        for document in rng.sample(range(1, 601), 100):
            corpus.put(document)

        corpus.assert_same_scores(100)


def test_search_ranks_best_first_and_newest_first_on_ties() -> None:
    index = Bm25Index()
    index.add(1, "the cat sat on the mat")
    index.add(2, "a dog")
    index.add(3, "cat cat cat")
    index.add(4, "the cat sat on the mat")
    index.add(5, "birds sing")

    found = index.search("cats", limit=2)

    assert [hit.document for hit in found] == [3, 4]
    assert [hit.document for hit in index.search("cats", limit=10)] == [3, 4, 1]
    assert found[1] == Hit(4, index.scores("cat")[1]), "4 ties with 1"


def test_adding_a_document_again_replaces_it() -> None:
    index = Bm25Index()
    index.add(1, "apples")
    index.add(1, "pears")

    assert len(index) == 1
    assert index.search("apples", limit=5) == []
    assert [hit.document for hit in index.search("pear", limit=5)] == [1]


def test_a_removed_document_is_gone_and_removing_it_again_says_so() -> None:
    index = Bm25Index()
    index.add(1, "apples")
    index.add(2, "pears")

    assert index.remove(1)
    assert not index.remove(1)
    assert 1 not in index
    assert index.scores("apples") == {}


def test_nothing_matches_an_empty_index_or_a_query_without_words() -> None:
    index = Bm25Index()
    assert index.search("anything", limit=5) == []
    index.add(1, "something")
    assert index.search("?! ...", limit=5) == []


def test_a_word_in_most_documents_still_ranks_them() -> None:
    index = Bm25Index()
    for document in range(1, 5):
        index.add(document, "common " * document)

    assert [hit.document for hit in index.search("common", limit=4)] == [4, 3, 2, 1]
    assert all(0 < hit.score < 1e-5 for hit in index.search("common", limit=4))
