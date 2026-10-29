# ruff: noqa: RUF001
# The expected strings are IPA symbols, confusable with ASCII by design.
from pathlib import Path

import pytest

from synthia.kernel.config import default_home
from synthia.models.catalogue import CMUDICT
from synthia.models.install import BYTES_PER_GB, Installer
from synthia.voice.phonemes import (
    PHONEMES,
    PUNCTUATION,
    Letters,
    Mark,
    PhonemeError,
    Phonemizer,
    Pronunciation,
    Pronunciations,
    Word,
    normalise,
    spell,
    to_misaki,
)

# Kokoro v1.0's vocabulary letters, copied from its tokenizer.json.
KOKORO_VOCABULARY = (
    ";:,.!?…AIOQSTWYᵊabcdefhijklmnopqrstuvwxyz"
    "ɑɐɒæβɔɕçɖðʤəɚɛɜɟɡɥɨɪʝɯɰŋɳɲɴøɸθœɹɾɻʁɽʂʃʈʧʊʋʌɣɤχʎʒʔˈˌːʰʲᵻ"
)
ARPABET = (
    *("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY"),
    *("UH", "UW", "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N"),
    *("NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"),
)
VOWELS = frozenset(ARPABET[:15])
SMALL = """\
# A test dictionary in the CMU format.
a AH0
a(2) EY1
ai AY1
bus B AH1 S
cat K AE1 T
dog D AO1 G
hello HH AH0 L OW1
hello(2) HH EH0 L OW1

i AY1
jump JH AH1 M P
nasa N AE1 S AH0 # abbrev
read R IY1 D
read(2) R EH1 D
us AH1 S
world W ER1 L D
"""


@pytest.fixture
def small() -> Pronunciations:
    return Pronunciations.parse(SMALL)


@pytest.mark.parametrize(
    ("phone", "symbol"),
    [
        ("AA0", "ɑ"),
        ("AE0", "æ"),
        ("AH0", "ə"),
        ("AH1", "ˈʌ"),
        ("AH2", "ˌʌ"),
        ("AO0", "ɔ"),
        ("AW0", "W"),
        ("AY0", "I"),
        ("EH0", "ɛ"),
        ("ER0", "əɹ"),
        ("ER1", "ˈɜɹ"),
        ("ER2", "ˌɜɹ"),
        ("EY0", "A"),
        ("IH0", "ɪ"),
        ("IY0", "i"),
        ("OW0", "O"),
        ("OW1", "ˈO"),
        ("OW2", "ˌO"),
        ("OY0", "Y"),
        ("UH0", "ʊ"),
        ("UW0", "u"),
        ("B", "b"),
        ("CH", "ʧ"),
        ("D", "d"),
        ("DH", "ð"),
        ("F", "f"),
        ("G", "ɡ"),
        ("HH", "h"),
        ("JH", "ʤ"),
        ("K", "k"),
        ("L", "l"),
        ("M", "m"),
        ("N", "n"),
        ("NG", "ŋ"),
        ("P", "p"),
        ("R", "ɹ"),
        ("S", "s"),
        ("SH", "ʃ"),
        ("T", "t"),
        ("TH", "θ"),
        ("V", "v"),
        ("W", "w"),
        ("Y", "j"),
        ("Z", "z"),
        ("ZH", "ʒ"),
    ],
)
def test_to_misaki_maps_each_phone(phone: str, symbol: str) -> None:
    assert to_misaki((phone,)) == symbol


def test_to_misaki_covers_all_39_phones_and_emits_only_phonemes() -> None:
    assert len(set(ARPABET)) == 39
    for phone in ARPABET:
        forms = [f"{phone}{d}" for d in "012"] if phone in VOWELS else [phone]
        for form in forms:
            assert set(to_misaki((form,))) <= PHONEMES, form


def test_phonemes_and_punctuation_are_in_kokoros_vocabulary() -> None:
    assert (PHONEMES | PUNCTUATION).issubset(KOKORO_VOCABULARY)


@pytest.mark.parametrize(
    ("pronunciation", "misaki"),
    [
        (("JH", "AH1", "M", "P"), "ʤˈʌmp"),
        (("W", "ER1", "L", "D"), "wˈɜɹld"),
        (("HH", "AH0", "L", "OW1"), "həlˈO"),
        ((), ""),
    ],
)
def test_to_misaki_puts_each_stress_mark_before_its_vowel(
    pronunciation: Pronunciation, misaki: str
) -> None:
    assert to_misaki(pronunciation) == misaki


@pytest.mark.parametrize("phone", ["AH", "QQ", "b", "AH3"])
def test_to_misaki_refuses_an_unknown_phone_or_an_unstressed_vowel(phone: str) -> None:
    with pytest.raises(ValueError, match=rf"not an ARPAbet phone .*'{phone}'"):
        to_misaki(("HH", phone))


def test_spell_says_each_letter_name_and_skips_the_rest() -> None:
    assert spell("w-x") == ("D", "AH1", "B", "AH0", "L", "Y", "UW0", "EH1", "K", "S")
    assert spell("Qz") == ("K", "Y", "UW1", "Z", "IY1")
    with pytest.raises(ValueError, match=r"no letter to spell in '123'"):
        spell("123")


def test_pronunciations_parse_keeps_every_variant_in_file_order(
    small: Pronunciations,
) -> None:
    assert small.variants("read") == (("R", "IY1", "D"), ("R", "EH1", "D"))
    assert small.first("READ") == ("R", "IY1", "D")
    assert small.first("nasa") == ("N", "AE1", "S", "AH0")
    assert small.first("zork") is None
    assert small.variants("zork") == ()
    assert len(small) == 12
    assert sum(len(small.variants(word)) for word in small) == 15


def test_pronunciations_parse_refuses_a_word_without_phones() -> None:
    with pytest.raises(ValueError, match=r"line 3: 'world' has no phones"):
        Pronunciations.parse("hello HH AH0 L OW1\n# comment\nworld # no phones\n")


def test_pronunciations_load_reads_the_installed_file(tmp_path: Path) -> None:
    (tmp_path / CMUDICT.dictionary.name).write_text(SMALL, encoding="utf-8")

    assert Pronunciations.load(tmp_path, CMUDICT).first("world") == (
        "W",
        "ER1",
        "L",
        "D",
    )


def test_pronunciations_load_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PhonemeError, match=r"pronouncing dictionary cmudict"):
        Pronunciations.load(tmp_path, CMUDICT)


def test_pronunciations_load_refuses_a_file_that_is_not_utf8(tmp_path: Path) -> None:
    (tmp_path / CMUDICT.dictionary.name).write_bytes(b"caf\xe9 K AE0 F EY1\n")

    with pytest.raises(PhonemeError, match=r"pronouncing dictionary cmudict"):
        Pronunciations.load(tmp_path, CMUDICT)


def said(text: str) -> str:
    rendered: list[str] = []
    for token in normalise(text):
        match token:
            case Word(text=word):
                rendered.append(word)
            case Letters(text=letters):
                rendered.append(f"[{letters}]")
            case Mark(symbol=symbol):
                rendered.append(symbol)
    return " ".join(rendered)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("it\N{RIGHT SINGLE QUOTATION MARK}s", "it's"),
        ("Hello, world!", "Hello , world !"),
        ("Why; because: yes.", "Why ; because : yes ."),
        ("Wait... what\N{HORIZONTAL ELLIPSIS}", "Wait … what …"),
        ("yes - no", "yes , no"),
        ("yes\N{EN DASH}no", "yes , no"),
        ("yes \N{EM DASH} no", "yes , no"),
        ('"Quoted" (aside) [x] {y} <z> *w* #1 @me', "Quoted aside x y z w one me"),
        ("salt & pepper", "salt and pepper"),
        ("AT&T", "AT and T"),
        ("Mr. Smith", "mister Smith"),
        ("MRS. Jones", "missus Jones"),
        ("Ms. Lee", "ms Lee"),
        ("the U.S. Army", "the [US] Army"),
        ("U.S.A.", "[USA]"),
        ("foo -- bar", "foo , bar"),
        ("-$5", "minus five dollars"),
        ("\N{MINUS SIGN}5", "minus five"),
        ("1990s", "nineteen nineties"),
        ("1990's", "nineteen nineties"),
        ("2000s", "two thousands"),
        ("90s", "nineties"),
        ("'60s", "sixties"),
        ("Dr. Who", "doctor Who"),
        ("cats vs. dogs", "cats versus dogs"),
        ("cats VS dogs", "cats versus dogs"),
        ("pens, etc., too", "pens , et cetera , too"),
        ("e.g., this", "for example , this"),
        ("i.e. that", "that is that"),
        ("at 9 a.m.", "at nine [am]"),
        ("5 P.M.", "five [pm]"),
        ("0", "zero"),
        ("7", "seven"),
        ("13", "thirteen"),
        ("40", "forty"),
        ("99", "ninety nine"),
        ("100", "one hundred"),
        ("101", "one hundred one"),
        ("1,234", "one thousand two hundred thirty four"),
        ("12345", "twelve thousand three hundred forty five"),
        ("1,000,000", "one million"),
        ("1,000,001", "one million one"),
        (
            "999,999,999,999",
            (
                "nine hundred ninety nine billion nine hundred ninety nine million "
                "nine hundred ninety nine thousand nine hundred ninety nine"
            ),
        ),
        ("1000000000000", " ".join(["one", *["zero"] * 12])),
        ("-5", "minus five"),
        ("007", "zero zero seven"),
        ("3.14", "three point one four"),
        ("-0.5", "minus zero point five"),
        ("1st", "first"),
        ("2nd", "second"),
        ("3rd", "third"),
        ("4th", "fourth"),
        ("11th", "eleventh"),
        ("12th", "twelfth"),
        ("13th", "thirteenth"),
        ("20th", "twentieth"),
        ("21st", "twenty first"),
        ("22nd", "twenty second"),
        ("101st", "one hundred first"),
        ("1000th", "one thousandth"),
        ("1100", "eleven hundred"),
        ("1900", "nineteen hundred"),
        ("1905", "nineteen oh five"),
        ("1999", "nineteen ninety nine"),
        ("2000", "two thousand"),
        ("2005", "two thousand five"),
        ("2010", "twenty ten"),
        ("2026", "twenty twenty six"),
        ("2099", "twenty ninety nine"),
        ("1099", "one thousand ninety nine"),
        ("2100", "two thousand one hundred"),
        ("in 1999.", "in nineteen ninety nine ."),
        ("4:30", "four thirty"),
        ("4:05", "four oh five"),
        ("4:00", "four o'clock"),
        ("23:59", "twenty three fifty nine"),
        ("$5", "five dollars"),
        ("$1", "one dollar"),
        ("$1.50", "one dollar fifty cents"),
        ("$2.01", "two dollars one cent"),
        ("$5.00", "five dollars"),
        ("$1.5", "one point five dollars"),
        ("$1,000", "one thousand dollars"),
        ("50%", "fifty percent"),
        ("2.5%", "two point five percent"),
        ("2026-10-29", "October twenty ninth , twenty twenty six"),
        ("2000-01-01", "January first , two thousand"),
        ("V5a", "[V] five [a]"),
        ("mp3", "mp three"),
        ("3D", "three [D]"),
        ("state-of-the-art", "state of the art"),
        ("x-ray", "x ray"),
    ],
)
def test_normalise_reads_each_form_as_words(text: str, expected: str) -> None:
    assert said(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("?!", "? !"),
        ("1,2,3", "one , two , three"),
        ("$", ""),
        ("'", ""),
        ("--", ","),
        ("hi 👋", "hi"),
        ("12:75", "twelve : seventy five"),
        ("24:00", "twenty four : zero zero"),
        ("2026-13-01", "twenty twenty six thirteen zero one"),
        ("café", "cafe"),
        ("日本", ""),
        ("0999", "zero nine nine nine"),
        ("-hello", "hello"),
    ],
)
def test_normalise_survives_input_the_rules_did_not_anticipate(
    text: str, expected: str
) -> None:
    assert said(text) == expected


def test_normalise_reads_a_number_too_long_for_int_digit_by_digit() -> None:
    assert said("9" * 5000) == " ".join(["nine"] * 5000)
    assert said(f"{'9' * 5000}th") == " ".join([*["nine"] * 5000, "th"])


@pytest.mark.parametrize(
    ("text", "misaki"),
    [
        ("Hello, world.", "həlˈO, wˈɜɹld."),
        ("hello?! world", "həlˈO?! wˈɜɹld"),
        ("...hello", "həlˈO"),
        ("?!", ""),
        ("", ""),
        ("AI", "ˈAˈI"),
        ("US", "jˈuˈɛs"),
        ("us", "ˈʌs"),
        ("BUS", "bˈijˈuˈɛs"),
        ("NASA", "nˈæsə"),
        ("HTTP", "ˈAʧtˈitˈipˈi"),
        ("I", "ˈI"),
        ("A", "ə"),
        ("read", "ɹˈid"),
        ("jump", "ʤˈʌmp"),
        ("cat's", "kˈæts"),
        ("bus's", "bˈʌsɪz"),
        ("dog's", "dˈɔɡz"),
        ("NASA's", "nˈæsəz"),
        ("AI's", "ˈAˈIz"),
        ("a.m.", "ˈAˈɛm"),
        ("xq", "ˈɛkskjˈu"),
    ],
)
def test_phonemizer_reads_known_words_and_spells_the_rest(
    small: Pronunciations, text: str, misaki: str
) -> None:
    assert Phonemizer(small).phonemes(text) == misaki


def test_phonemizer_asks_the_guess_for_unknown_lower_case_words(
    small: Pronunciations,
) -> None:
    asked: list[str] = []

    def guess(word: str) -> Pronunciation:
        asked.append(word)
        return ("Z", "AO1", "R", "K")

    phonemizer = Phonemizer(small, guess)

    assert (
        phonemizer.phonemes("Zork hello HTTP zork's")
        == "zˈɔɹk həlˈO ˈAʧtˈitˈipˈi zˈɔɹk"
    )
    assert asked == ["zork", "zork's"]


def test_phonemizer_refuses_a_guess_with_a_phone_misaki_lacks(
    small: Pronunciations,
) -> None:
    with pytest.raises(ValueError, match=r"not an ARPAbet phone"):
        Phonemizer(small, lambda _: ("AH",)).phonemes("zork")


@pytest.fixture(scope="module")
def installed() -> Pronunciations:
    installer = Installer(default_home(), BYTES_PER_GB)
    if not installer.installed(CMUDICT):
        pytest.skip("cmudict is not installed (synthia models install cmudict)")
    return Pronunciations.load(installer.path_of(CMUDICT), CMUDICT)


def test_the_installed_dictionary_reads_hello_world(installed: Pronunciations) -> None:
    assert Phonemizer(installed).phonemes("Hello, world.") == "həlˈO, wˈɜɹld."


def test_every_word_the_normaliser_writes_is_in_the_installed_dictionary(
    installed: Pronunciations,
) -> None:
    forms = [
        "Mr. Mrs. Ms. Dr. vs. etc. e.g. i.e. & -1 2.5 50% $1 $2.01",
        *(f"{n} {n}th {n}0s 1{n:02d}0s" for n in range(1, 101)),
        *(f"2026-{month:02d}-01" for month in range(1, 13)),
        "1,000,000,000 1905 2005 4:00 4:05",
    ]
    words = {
        token.text
        for form in forms
        for token in normalise(form)
        if isinstance(token, Word)
    }
    assert {word for word in words if installed.first(word) is None} == set()


def test_every_installed_pronunciation_converts_to_phonemes(
    installed: Pronunciations,
) -> None:
    for word in installed:
        for pronunciation in installed.variants(word):
            assert set(to_misaki(pronunciation)) <= PHONEMES, word


def test_the_installed_dictionary_keeps_every_word_and_variant(
    installed: Pronunciations,
) -> None:
    assert len(installed) == 126_052
    assert sum(len(installed.variants(word)) for word in installed) == 135_166
