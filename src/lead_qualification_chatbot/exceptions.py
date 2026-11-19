class AppError(Exception):
    """Base class for all application errors."""


class InvalidInputError(AppError):
    """User input failed validation. `message` is safe to show the visitor."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


class DataLoadError(AppError):
    """A CSV is missing, malformed or fails schema checks."""


class LLMProviderError(AppError):
    """LLM or embedding provider failed after retries."""


class ConfigError(AppError):
    """Missing or invalid configuration (e.g. no API key for the chosen provider)."""
