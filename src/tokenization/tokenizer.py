"""Tokenizer abstraction with subword-to-word alignment support."""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
from transformers import AutoTokenizer
from src.tokenization.alignment import TokenInfo, WordInfo, extract_words_with_spans, align_tokens_to_words


class TokenizerWrapper:
    """Wrapper around HuggingFace AutoTokenizer providing word and token level alignment."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.special_token_ids = set(self.tokenizer.all_special_ids)

    def tokenize_and_align(self, text: str) -> Tuple[List[TokenInfo], List[WordInfo], Any]:
        """
        Tokenize input text, preserve subword-to-word mapping and return
        aligned token list, word list, and raw tokenizer encoding.
        """
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors=None,
            add_special_tokens=True
        )

        input_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]

        # Extract base words from text
        words = extract_words_with_spans(text)

        tokens: List[TokenInfo] = []
        for idx, (tid, (start, end)) in enumerate(zip(input_ids, offset_mapping)):
            is_special = tid in self.special_token_ids or (start == 0 and end == 0 and idx > 0 and len(text) > 0)
            token_str = self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)

            tokens.append(TokenInfo(
                token_index=idx,
                token_id=tid,
                token=token_str,
                start_char=start,
                end_char=end,
                word_id=None,
                is_special=is_special
            ))

        tokens, words = align_tokens_to_words(tokens, words)
        return tokens, words, encoding

    @staticmethod
    def remove_word_span(text: str, word: WordInfo) -> str:
        """
        Remove a word span cleanly from text and clean up resulting spaces.
        Example: 'That cat chased' -> 'That chased'
        """
        before = text[:word.start_char]
        after = text[word.end_char:]
        
        # If removing left adjacent space or double spaces
        res = (before + after).strip()
        # Collapse multiple spaces into single space, but preserve newlines
        import re
        res = re.sub(r'[ \t]+', ' ', res)
        return res

    @staticmethod
    def remove_word_spans(text: str, words_to_remove: List[WordInfo]) -> str:
        """
        Remove multiple words by span in reverse order (to avoid offset shifting).
        """
        sorted_words = sorted(words_to_remove, key=lambda w: w.start_char, reverse=True)
        res = text
        for w in sorted_words:
            res = res[:w.start_char] + res[w.end_char:]
        import re
        res = re.sub(r'[ \t]+', ' ', res)
        # Clean spacing before punctuation
        res = re.sub(r'\s+([.,!?;:])', r'\1', res)
        return res.strip()
