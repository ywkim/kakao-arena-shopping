import abc
from typing import List, Iterable, Sequence
import unicodedata

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


class CharacterTokenizer(Tokenizer):
    """
    A `CharacterTokenizer` splits a sentence into character tokens.

    Args:
      decompose: Whether to normalize the sentence to NFD before splitting.
    """

    def __init__(self, decompose: bool = False) -> None:
        self._decompose = decompose

    def tokenize(self, sentence: str) -> List[Token]:
        sentence = sentence.strip()
        if self._decompose:
            # NFC -> NFD
            sentence = unicodedata.normalize("NFD", sentence)
        tokens = [Token(c) for c in sentence]
        return tokens

    def detokenize(self, tokens: Iterable[str]) -> str:
        sentence = "".join(tokens)
        if self._decompose:
            # NFD -> NFC
            sentence = unicodedata.normalize("NFC", sentence)
        return sentence


class WhitespaceTokenizer(Tokenizer):
    """
    A `WhitespaceTokenizer` splits a sentence into word tokens.

    Args:
      do_lower_case: Whether to lower case the input.
    """

    def __init__(self, do_lower_case: bool = False) -> None:
        self._do_lower_case = do_lower_case

    def tokenize(self, sentence: str) -> List[Token]:
        words = sentence.strip().split()
        tokens = [Token(self._process_token(word)) for word in words]
        return tokens

    def _process_token(self, token):
        if self._do_lower_case:
            token = token.lower()
            token = self._run_strip_accents(token)
        return token

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def detokenize(self, tokens: Iterable[str]) -> str:
        sentence = " ".join(tokens)
        if self._do_lower_case:
            sentence = unicodedata.normalize("NFC", sentence)
        return sentence


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
