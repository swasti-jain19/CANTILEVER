import re
import string

__all__ = ["clean_text"]

_punct_table = str.maketrans("", "", string.punctuation)

def clean_text(text: str) -> str:
    """Basic, fast text cleaning suitable for TF-IDF.

    - lowercases

    - removes URLs, mentions, hashtags, digits

    - strips punctuation and extra spaces

    """

    if text is None:

        return ""

    x = text.lower()

    x = re.sub(r"https?://\S+|www\.\S+", " ", x)

    x = re.sub(r"[@#]\w+", " ", x)

    x = re.sub(r"\d+", " ", x)

    x = x.translate(_punct_table)

    x = re.sub(r"\s+", " ", x).strip()

    return x

