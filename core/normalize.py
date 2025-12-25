from __future__ import annotations
import re
import unicodedata

SOFT_HYPHEN = "\u00AD"
NBSP = "\u00A0"
ZWSP = "\u200B"
BOM = "\uFEFF"


def to_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def replace_nbsp(text: str) -> str:
    return text.replace(NBSP, " ")


def remove_zero_width_and_bom(text: str) -> str:
    return text.replace(ZWSP, "").replace(BOM, "")


def normalize_newlines(text: str) -> str:
    re = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in re.split("\n"))
    return text


def collapse_spaces(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s


def fix_hyphenation(s: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", s)


def collapse_blank_lines(s: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", s)


def strip_soft_hyphen(s: str) -> str:
    return s.replace(SOFT_HYPHEN, "")


def normalize_text(raw: str) -> str:
    s = raw
    s = to_nfc(s)
    s = replace_nbsp(s)
    s = remove_zero_width_and_bom(s)
    s = normalize_newlines(s)
    s = strip_soft_hyphen(s)
    s = collapse_spaces(s)
    s = fix_hyphenation(s)
    s = collapse_blank_lines(s)
    return s.strip()
