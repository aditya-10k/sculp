"""Token and Word alignment data structures and utilities."""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TokenInfo:
    """Represents a model token with alignment metadata."""
    token_index: int
    token_id: int
    token: str
    start_char: int
    end_char: int
    word_id: Optional[int]
    is_special: bool = False

    def to_dict(self) -> dict:
        return {
            "token_index": self.token_index,
            "token_id": self.token_id,
            "token": self.token,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "word_id": self.word_id,
            "is_special": self.is_special
        }


@dataclass
class WordInfo:
    """Represents a natural language word/token with character offsets and subtoken indices."""
    word_id: int
    word: str
    start_char: int
    end_char: int
    subtoken_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "word_id": self.word_id,
            "word": self.word,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "subtoken_indices": self.subtoken_indices
        }


def extract_words_with_spans(text: str) -> List[WordInfo]:
    """
    Split text into natural words/punctuation tokens with exact character spans.
    Preserves exact character indices from original text.
    """
    # Matches alphanumeric words, or non-whitespace sequences/punctuation
    pattern = re.compile(r'\w+|[^\w\s]')
    words: List[WordInfo] = []
    word_id = 0
    for match in pattern.finditer(text):
        word_str = match.group(0)
        start, end = match.span()
        words.append(WordInfo(
            word_id=word_id,
            word=word_str,
            start_char=start,
            end_char=end,
            subtoken_indices=[]
        ))
        word_id += 1
    return words


def align_tokens_to_words(tokens: List[TokenInfo], words: List[WordInfo]) -> Tuple[List[TokenInfo], List[WordInfo]]:
    """
    Align model subtokens to natural language words based on character spans.
    Updates token.word_id and word.subtoken_indices.
    """
    # Create word lookup by interval
    for token in tokens:
        if token.is_special:
            token.word_id = None
            continue

        matched_word_id = None
        for word in words:
            # Overlap check
            if not (token.end_char <= word.start_char or token.start_char >= word.end_char):
                matched_word_id = word.word_id
                break

        token.word_id = matched_word_id
        if matched_word_id is not None:
            words[matched_word_id].subtoken_indices.append(token.token_index)

    return tokens, words
