from typing import Dict, List


def build_vocab(alphabet: str) -> Dict[str, int]:
    """
    Create a mapping from characters to integer indices, adding the special CTC
    blank token at the end. The alphabet string comes from config (e.g., A-Z,
    digits, punctuation, and space).
    """
    labels = list(alphabet)
    stoi = {ch: idx for idx, ch in enumerate(labels)}
    stoi["<BLANK>"] = len(labels)
    return stoi


def index_to_char(labels: str) -> List[str]:
    """
    Build the reverse lookup list (index -> character) used during decoding.
    The blank token is appended to keep indices aligned with the model output.
    """
    chars = list(labels)
    chars.append("<BLANK>")
    return chars
