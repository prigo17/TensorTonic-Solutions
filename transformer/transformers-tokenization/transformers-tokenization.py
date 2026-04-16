import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token

        words = []

        for text in texts:
            tokens = text.split()

            for token in tokens:
                words.append(token)


        words.sort()

        
        for word in words:
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.word_to_id)

                self.id_to_word[len(self.word_to_id) - 1] = word


        self.vocab_size = len(self.word_to_id)

        return
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        tokens = text.split()

        ans = []

        for token in tokens:
            lt = token.lower()
            ans.append(self.word_to_id[lt] if lt in self.word_to_id else self.word_to_id[self.unk_token])

        return ans
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """

        tokens = []

        for id in ids:
            tokens.append(self.id_to_word[id] if id in self.id_to_word else self.id_to_word[1])

        return " ".join(tokens)
