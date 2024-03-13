import re
import string


def normalize(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        text = text.replace("_", " ")  # consider underscores spaces
        exclude = set(string.punctuation)
        exclude.remove("/")  # don't remove slashes
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def get_tokens(s: str) -> list[str]:
    """Gets normalized tokens from the given string."""
    if not s:
        return []
    # Split on all whitespace and slashes
    return re.split(r"[\s/]+", normalize(s))
