import dataclasses
from typing import get_args

import pytest

import anyverbatim
from anyverbatim.text import numbers


def test_quantity_tokens_is_part_of_the_library_surface() -> None:
    assert {"quantity_tokens", "QToken", "QKind"} <= set(anyverbatim.__all__)
    assert anyverbatim.quantity_tokens is numbers.quantity_tokens
    assert anyverbatim.QToken is numbers.QToken
    assert anyverbatim.QKind is numbers.QKind


def test_the_token_shape_is_pinned() -> None:
    assert [(f.name, f.type) for f in dataclasses.fields(anyverbatim.QToken)] == [
        ("char_start", int),
        ("char_end", int),
        ("surface", str),
        ("value_itn", str),
        ("kind", numbers.QKind),
    ]
    assert get_args(anyverbatim.QKind) == ("number", "currency", "percent", "ticker", "cjk_quantity")


def test_tokens_carry_offsets_surface_value_and_kind() -> None:
    text = "SPY 涨了 3.5%，一百五十期，$1.5B"
    assert anyverbatim.quantity_tokens(text) == [
        anyverbatim.QToken(0, 3, "SPY", "SPY", "ticker"),
        anyverbatim.QToken(7, 11, "3.5%", "3.5%", "percent"),
        anyverbatim.QToken(12, 17, "一百五十期", "150期", "cjk_quantity"),
        anyverbatim.QToken(18, 23, "$1.5B", "$1500000000", "currency"),
    ]


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("2023,2024", [(0, 4, "2023", "2023", "number"), (5, 9, "2024", "2024", "number")]),
        ("3.35SMH", [(0, 4, "3.35", "3.35", "number"), (4, 7, "SMH", "SMH", "ticker")]),
        ("twenty-five", [(0, 11, "twenty-five", "25", "number")]),
        ("10BN", [(0, 4, "10BN", "10000000000", "number")]),
        ("三万一个月", [(0, 2, "三万", "30000", "cjk_quantity"), (2, 5, "一个月", "1个月", "cjk_quantity")]),
        ("三家公司", [(0, 2, "三家", "3家", "cjk_quantity")]),
    ],
)
def test_pinned_token_examples(text: str, tokens: list[tuple[int, int, str, str, str]]) -> None:
    assert anyverbatim.quantity_tokens(text) == [anyverbatim.QToken(*token) for token in tokens]
