"""Unit tests for tokenizer and subword-word alignment."""

import pytest
from src.tokenization.tokenizer import TokenizerWrapper
from src.tokenization.alignment import extract_words_with_spans


def test_word_span_extraction():
    text = "That cat chased the mouse across the garden."
    words = extract_words_with_spans(text)
    word_strs = [w.word for w in words]
    assert word_strs == ["That", "cat", "chased", "the", "mouse", "across", "the", "garden", "."]
    assert words[0].start_char == 0
    assert words[0].end_char == 4
    assert words[1].word == "cat"
    assert words[1].start_char == 5
    assert words[1].end_char == 8


def test_token_alignment_with_bert():
    tokenizer = TokenizerWrapper("bert-base-uncased")
    text = "That cat chased the mouse across the garden."
    tokens, words, encoding = tokenizer.tokenize_and_align(text)

    # Special tokens [CLS] and [SEP]
    assert tokens[0].is_special
    assert tokens[0].token == "[CLS]"
    assert tokens[-1].is_special
    assert tokens[-1].token == "[SEP]"

    # Non-special tokens should be assigned to words
    non_special = [t for t in tokens if not t.is_special]
    for t in non_special:
        assert t.word_id is not None
        assert 0 <= t.word_id < len(words)

    # Check words have subtoken indices
    for w in words:
        assert len(w.subtoken_indices) >= 1


def test_subword_alignment():
    tokenizer = TokenizerWrapper("bert-base-uncased")
    text = "The antigravity device is performing lexicalization."
    tokens, words, _ = tokenizer.tokenize_and_align(text)

    # Find the word "antigravity"
    antigravity_words = [w for w in words if w.word == "antigravity"]
    assert len(antigravity_words) == 1
    target_word = antigravity_words[0]

    # In bert-base-uncased, 'antigravity' is split into multiple subtokens: 'anti', '##gra', '##vity'
    subtoken_indices = target_word.subtoken_indices
    assert len(subtoken_indices) >= 2
    for idx in subtoken_indices:
        assert tokens[idx].word_id == target_word.word_id


def test_second_sanity_sentence():
    tokenizer = TokenizerWrapper("bert-base-uncased")
    text = "The cat chased the mouse across the garden."
    tokens, words, _ = tokenizer.tokenize_and_align(text)
    non_special = [t for t in tokens if not t.is_special]
    assert len(non_special) >= len(words)
    assert words[0].word == "The"
    assert words[1].word == "cat"
