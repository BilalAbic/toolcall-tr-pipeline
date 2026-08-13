from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from toolcall_tr.hashing import CanonicalizationError, canonical_bytes, sha256_jcs


@given(
    st.dictionaries(
        st.text(min_size=1), st.integers(min_value=-(2**53) + 1, max_value=(2**53) - 1), max_size=20
    )
)
def test_object_key_order_never_changes_jcs_hash(value: dict[str, int]) -> None:
    reversed_value = dict(reversed(list(value.items())))
    assert canonical_bytes(value) == canonical_bytes(reversed_value)
    assert sha256_jcs(value) == sha256_jcs(reversed_value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_not_hashable(value: float) -> None:
    with pytest.raises(CanonicalizationError):
        sha256_jcs({"value": value})
