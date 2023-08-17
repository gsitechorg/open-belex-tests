r"""
By Dylon Edwards
"""

import numpy as np

import hypothesis
from hypothesis import given

from open_belex.diri.half_bank import DIRI
from open_belex.literal import RL, VR, belex_apl
from open_belex.utils.example_utils import convert_to_bool

from open_belex_tests.utils import parameterized_belex_test, vr_strategy


@belex_apl
def sb_from_inv_sb(Belex, dst: VR, src: VR):
    RL[::] <= ~src()
    dst[::] <= RL()


@hypothesis.settings(max_examples=3, deadline=None)
@given(data=vr_strategy())
@parameterized_belex_test
def test_sb_from_inv_sb(diri: DIRI, data: np.ndarray) -> int:
    dst = 0
    src = 1

    diri.hb[src, ::, ::] = convert_to_bool(data)
    expected_value = ~diri.hb[src]

    sb_from_inv_sb(dst, src)
    assert np.array_equal(expected_value, diri.hb[dst])
