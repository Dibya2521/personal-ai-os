from pathlib import Path

import ci_annotations
import pytest

REPORT = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
<testcase classname="tests.a.test_x" name="test_passes" time="0.1"/>
<testcase classname="tests.a.test_x" name="test_fails[a,b]" time="0.2">
<failure message="AssertionError: 100% wrong&#10;second line">long traceback</failure>
</testcase>
<testcase classname="tests.b.test_y" name="test_errors" time="0.3">
<error message="">RuntimeError: fixture broke
more</error>
</testcase>
<testcase classname="tests.b.test_y" name="test_skipped">
<skipped message="posix"/></testcase>
</testsuite></testsuites>
"""


def test_failures_lists_each_failed_or_errored_test_with_its_first_line(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.xml"
    report.write_text(REPORT, encoding="utf-8")

    found = list(ci_annotations.failures(report))

    assert found == [
        ("tests.a.test_x::test_fails[a,b]", "AssertionError: 100% wrong"),
        ("tests.b.test_y::test_errors", "RuntimeError: fixture broke"),
    ]


def test_an_annotation_escapes_what_github_would_read_as_a_command() -> None:
    line = ci_annotations.annotation("tests.a::test_fails[a,b]", "100% bad\nnext")

    assert line == "::error title=tests.a%3A%3Atest_fails[a%2Cb]::100%25 bad%0Anext"


def test_a_long_message_is_cut_to_the_limit(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    long = "x" * (ci_annotations.MESSAGE_LIMIT * 2)
    report.write_text(
        f'<testsuite><testcase classname="c" name="n"><failure message="{long}"/>'
        "</testcase></testsuite>",
        encoding="utf-8",
    )

    ((_, message),) = ci_annotations.failures(report)

    assert len(message) == ci_annotations.MESSAGE_LIMIT


def test_main_prints_one_annotation_per_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "report.xml"
    report.write_text(REPORT, encoding="utf-8")

    assert ci_annotations.main([str(report)]) == 0

    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 2
    assert all(line.startswith("::error title=") for line in printed)


def test_main_says_so_when_pytest_left_no_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ci_annotations.main([str(tmp_path / "absent.xml")]) == 0

    assert "no report" in capsys.readouterr().out
