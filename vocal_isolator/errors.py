"""Errors shown directly to the person using the app."""


class VocalError(RuntimeError):
    """A failure with a message that is already safe to display."""
