"""Domain-specific exceptions raised by the backend."""


class MatSciAgentError(Exception):
    """Base class for recoverable backend errors."""


class MaterialNotFoundError(MatSciAgentError):
    """Raised when a material lookup does not produce a result."""
