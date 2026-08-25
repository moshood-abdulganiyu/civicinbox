# tests/test_pipeline.py
from app.core.preprocessing import normalize_text


def test_normalize_text_whitespace_and_case():
    raw = "  My Scholarship Application   Has Not Been Reviewed!!  \n"
    expected = "my scholarship application has not been reviewed!"
    assert normalize_text(raw) == expected


def test_normalize_text_internal_newlines_collapse():
    raw = "HELLO\n\n\nWorld"
    expected = "hello world"
    assert normalize_text(raw) == expected


def test_normalize_text_empty_string():
    assert normalize_text("") == ""


def test_normalize_text_repeated_punctuation():
    assert normalize_text("Is this urgent???") == "is this urgent?"
    assert normalize_text("please help....") == "please help."