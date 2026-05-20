from __future__ import annotations

import io
from pathlib import Path
import tokenize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = PROJECT_ROOT / "src" / "ssf" / "bob_exhaustive.py"


def test_b_exhaustive_has_no_strategy_shortcuts():
    text = EVALUATOR_PATH.read_text(encoding="utf-8")
    token_stream = tokenize.generate_tokens(io.StringIO(text).readline)

    filtered_parts: list[str] = []
    for tok in token_stream:
        if tok.type in (
            tokenize.COMMENT,
            tokenize.STRING,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
            tokenize.ENCODING,
        ):
            continue
        filtered_parts.append(tok.string)

    executable_text = " ".join(filtered_parts)

    forbidden = [
        'strategy.get("name")',
        "expected_scores",
        "_matches_",
        "majority",
        "hybrid",
        "pyramid",
        "return 0.5 +",
        "return 0.625",
        "return 0.75",
        "return 1.0",
    ]

    for token in forbidden:
        assert token not in executable_text

