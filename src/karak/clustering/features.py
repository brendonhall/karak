"""Feature-token grammar shared by the stages and the numeric core."""

from __future__ import annotations

import re

_RATIO = re.compile(r"^\s*([^/()\s]+)\s*/\s*\(\s*([^+()\s]+)\s*\+\s*([^+()\s]+)\s*\)\s*$")


def parse_feature(token: str) -> tuple[str, tuple[str, ...]]:
    """One feature token: a channel name, ``BSE`` or a ratio ``A/(A+B)``."""
    token = token.strip()
    if token.upper() == "BSE":
        return ("bse", ())
    if "/" in token:
        m = _RATIO.match(token)
        if m is None or m.group(1) != m.group(2):
            raise ValueError(f"ratio feature must look like A/(A+B), got {token!r}")
        return ("ratio", (m.group(1), m.group(3)))
    if not token:
        raise ValueError("empty feature name")
    return ("channel", (token,))
