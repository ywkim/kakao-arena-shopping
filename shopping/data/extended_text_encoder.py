from tensor2tensor.data_generators import text_encoder


class ShoppingTokenTextEncoder(text_encoder.TextEncoder):
    """A `TokenEncoder` encode/decode string tokens to/from integer ids."""

    def __init__(self, tokenizer, vocabulary):
        super().__init__()
        self._tokenizer = tokenizer
        self._vocabulary = vocabulary

    def encode(self, s):
        """Converts a space-separated string of tokens to a list of ids."""
        tokens = [token.text for token in self._tokenizer.tokenize(s)]
        ret = self.encode_list(tokens)
        return ret

    def encode_list(self, tokens):
        return [self._vocabulary.token_to_id(tok) for tok in tokens]

    def decode(self, ids, strip_extraneous=False):
        """Transform a sequence of int ids into a human-readable string."""
        return self._tokenizer.detokenize(self.decode_list(ids))

    def decode_list(self, ids):
        """Transform a sequence of int ids into a their string versions."""
        seq = ids
        return [self._vocabulary.id_to_token(i) for i in seq]

    @property
    def vocab_size(self):
        return self._vocabulary.vocab_size
