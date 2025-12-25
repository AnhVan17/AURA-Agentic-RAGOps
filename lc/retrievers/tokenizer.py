import re, unicodedata

WORD = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()

def simple_vi_en_tokens(text: str):
    s = normalize_unicode(text)
    return WORD.findall(s)