from collections import OrderedDict
import logging

from typing import List, Optional

import sentencepiece as spm
from tensor2tensor.data_generators import text_encoder

logger = logging.getLogger(__name__)


class Vocabulary:
    """
    A `Vocaburary` maps a string token to an integer id.
    Initialize tokens from a list of tokens.
    The set of tokens in vocab_list should be unique.

    Args:
      vocab_list: A list of elements of the vocabulary.
      oov_token: If not None, every out-of-vocabulary token seen when
          encoding will be replaced by this string (which must be in vocab).
    """

    def __init__(self, vocab_list: List[str], oov_token: Optional[str] = None) -> None:
        self._oov_token = oov_token
        if oov_token and oov_token not in vocab_list:
            raise ValueError(f'OOV token "{oov_token}" must be in vocab.')
        self._id_to_token = dict(enumerate(vocab_list))
        # _token_to_id is the reverse of _id_to_token
        self._token_to_id = dict((v, k) for k, v in self._id_to_token.items())

    def token_to_id(self, token: str) -> int:
        if self._oov_token is not None:
            if token not in self._token_to_id:
                token = self._oov_token
        return self._token_to_id[token]

    def id_to_token(self, idx: int) -> str:
        return self._id_to_token[idx]

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    def store_to_file(self, filename: str) -> None:
        """
        Write vocab file to disk.

        Vocab files have one token per line. The file ends in a newline. Reserved
        tokens are written to the vocab file as well.

        Args:
          filename: Full path of the file to store the vocab to.
        """
        with open(filename, "w", encoding="utf-8") as f:
            for i in range(len(self._id_to_token)):
                f.write(self._id_to_token[i] + "\n")

    @classmethod
    def from_file(cls, filename: str, oov_token: Optional[str] = None) -> "Vocabulary":
        """Load vocab from a file.

        Args:
          filename: The file to load vocabulary from.
        """
        with open(filename, encoding="utf-8") as vocab_file:
            tokens = [token.strip() for token in vocab_file.readlines()]

        vocab = cls(tokens, oov_token)
        return vocab

    @classmethod
    def from_pretrained_glove(
            cls,
            embedding_path: str,
            oov_token: str,
            additional_trainable_tokens: Optional[List[str]] = None,
    ) -> "Vocabulary":
        with open(embedding_path) as f:
            vocab_list = [line.split()[0] for line in f]
        logger.info("Found %s word vectors.", len(vocab_list))

        if additional_trainable_tokens:
            vocab_list = additional_trainable_tokens + vocab_list

        if oov_token:
            vocab_list = [oov_token] + vocab_list

        vocab_list = text_encoder.RESERVED_TOKENS + vocab_list

        vocab = cls(vocab_list, oov_token=oov_token)
        return vocab

    @classmethod
    def from_sentence_piece(cls,
                            model_path: str,
                            oov_token: str,
                            remove_space_symbol: bool = False) -> "Vocabulary":
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)

        tokens = [sp.IdToPiece(i) for i in range(len(sp))]
        # Skip first 3 tokens: <unk>, <s>, </s>
        tokens = tokens[3:]

        if remove_space_symbol:
            # Preserve the order
            unescaped_tokens = map(lambda s: s.replace("▁", ""), tokens)
            tokens = OrderedDict(zip(unescaped_tokens, tokens))
            # Remove the empty token
            tokens = filter(None, tokens)
            tokens = list(tokens)

        # Prepend PAD, EOS, oov_token
        tokens = text_encoder.RESERVED_TOKENS + [oov_token] + tokens
        vocab = cls(vocab_list=tokens, oov_token=oov_token)
        return vocab
