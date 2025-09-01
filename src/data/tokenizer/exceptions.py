class TokenizerError(Exception):
    """Base tokenizer exception."""

class VocabLoadError(TokenizerError):
    pass

class TrainingError(TokenizerError):
    pass

class DecodeError(TokenizerError):
    pass

class EncodeError(TokenizerError):
    pass
