"""Token protection configuration and rules for progressive pruning."""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Set

NEGATION_TERMS: Set[str] = {
    "not", "never", "no", "without", "except", "neither", "nor",
    "cannot", "can't", "won't", "don't", "must", "mustn't", "only",
    "at least", "at most", "exactly", "none", "hardly", "barely"
}

DELIMITER_TERMS: Set[str] = {
    "---", "===", "###", "##", "#", "```", "'''", "\n", "\t"
}


@dataclass
class ProtectionConfig:
    protect_delimiters: bool = True
    protect_code_blocks: bool = True
    protect_negation: bool = False
    protect_numbers: bool = False
    protect_dates: bool = False

    def is_protected(self, word_text: str) -> bool:
        w_lower = word_text.strip().lower()

        if self.protect_delimiters and (w_lower in DELIMITER_TERMS or "\n" in word_text):
            return True

        if self.protect_code_blocks and ("```" in word_text or "'''" in word_text):
            return True

        if self.protect_negation and w_lower in NEGATION_TERMS:
            return True

        if self.protect_numbers:
            # Matches digits, percentages, currency, e.g. 5, 20%, $20,000, 15ms
            if re.match(r'^[$€£¥]?\d+([.,]\d+)?%?$', w_lower):
                return True

        if self.protect_dates:
            # Matches dates like 2026, 06-10-2026, 1990s
            if re.match(r'^\d{4}s?$|^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', w_lower):
                return True

        return False
