import abc
from typing import List, Iterable, Sequence

import sentencepiece as spm


class Token:
    def __init__(self, text: str) -> None:
        self.text = text

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


class Tokenizer(abc.ABC):
    """A `Tokenizer` splits a sentence into tokens."""

    @abc.abstractmethod
    def tokenize(self, sentence: str) -> Sequence[Token]:
        """Returns tokens for given sentence."""

    def detokenize(self, tokens: Iterable[str]) -> str:
        return NotImplemented


class SentencePieceTokenizer(Tokenizer):
    """
    A tokenizer that does the SentencePiece tokenization.

    Args:
      model_path: path to the SentencePiece model ("<model_name>.model")
      remove_space_symbol: Whether to remove the space symbol.
    """

    def __init__(self, model_file: str, remove_space_symbol: bool = False) -> None:
        self.should_remove_space_symbol = remove_space_symbol
        self._tokenizer = spm.SentencePieceProcessor()
        self._tokenizer.Load(model_file)

    def __deepcopy__(self, memo):
        # SentencePieceProcessor is not serializable
        return None

    def tokenize(self, sentence: str) -> List[Token]:
        tokens = self._tokenizer.EncodeAsPieces(sentence)
        if self.should_remove_space_symbol:
            tokens = map(self._remove_space_symbol, tokens)
            # Remove empty tokens
            tokens = filter(None, tokens)
        tokens = [Token(token) for token in tokens]
        return tokens

    def _remove_space_symbol(self, s: str) -> str:
        return s.replace("▁", "")

    def detokenize(self, tokens: Iterable[str]) -> str:
        if self.should_remove_space_symbol:
            return " ".join(tokens)
        return self._tokenizer.DecodePieces(tokens)
