"""Single source for the OnCo licence line."""

from __future__ import annotations

ATTRIBUTION = (
    "Data from OnCo (onco.cc), CC BY-NC 4.0; commercial use needs a licence."
)

LICENCE = "CC BY-NC 4.0"
SOURCE_URL = "https://onco.cc"
REPO_URL = "https://github.com/judegomila/OnCo"


class MissingAttributionError(ValueError):
    """Raised when an export is missing the required attribution line."""


def assert_attribution(text: str) -> None:
    if ATTRIBUTION not in text and "Data from OnCo (onco.cc)" not in text:
        raise MissingAttributionError(
            "OnCo-derived output must include the CC BY-NC 4.0 attribution line."
        )
