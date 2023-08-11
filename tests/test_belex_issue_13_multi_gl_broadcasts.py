r"""By Dylon Edwards

Copyright 2019 - 2023 GSI Technology, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np

import hypothesis
from hypothesis import given

import open_belex.common.mask
from open_belex.diri.half_bank import DIRI
from open_belex.literal import GL, NOOP, RL, VR, Mask, apl_commands, belex_apl
from open_belex.utils.example_utils import convert_to_bool

from open_belex_tests.utils import (parameterized_belex_test, u16_strategy,
                                    vr_strategy)


@belex_apl
def gl_from_2_masks(Belex, source: VR, mask1: Mask, mask2: Mask):
    RL[::] <= source()
    with apl_commands():
        GL[mask1] <= RL()
        GL[mask2] <= RL()
    NOOP()


@belex_apl
def gl_from_3_masks(Belex, source: VR, mask1: Mask, mask2: Mask, mask3: Mask):
    RL[::] <= source()
    with apl_commands():
        GL[mask1] <= RL()
        GL[mask2] <= RL()
        GL[mask3] <= RL()
    NOOP()


@belex_apl
def gl_from_4_masks(Belex, source: VR, mask1: Mask, mask2: Mask, mask3: Mask, mask4: Mask):
    RL[::] <= source()
    with apl_commands():
        GL[mask1] <= RL()
        GL[mask2] <= RL()
        GL[mask3] <= RL()
        GL[mask4] <= RL()
    NOOP()


def expected_value_for(diri: DIRI, sections: int) -> np.ndarray:
    mask = open_belex.common.mask.Mask(f"0x{sections:04X}")
    expected_value = diri.RL()[::, list(mask)].all(axis=1)
    return expected_value


@hypothesis.settings(max_examples=5, deadline=None)
@given(source_vr=vr_strategy(),
       mask1=u16_strategy(),
       mask2=u16_strategy(),
       mask3=u16_strategy(),
       mask4=u16_strategy())
@parameterized_belex_test
def test_multi_gl_broadcasts(diri: DIRI,
                             source_vr: np.ndarray,
                             mask1: int,
                             mask2: int,
                             mask3: int,
                             mask4: int) -> int:

    source = 0
    diri.hb[source, ::, ::] = convert_to_bool(source_vr)

    gl_from_2_masks(source, mask1, mask2)
    assert np.array_equal(diri.GL, expected_value_for(diri, mask1 | mask2))

    gl_from_3_masks(source, mask1, mask2, mask3)
    assert np.array_equal(diri.GL,
                          expected_value_for(diri, mask1 | mask2 | mask3))

    gl_from_4_masks(source, mask1, mask2, mask3, mask4)
    assert np.array_equal(diri.GL,
                          expected_value_for(diri, mask1 | mask2 | mask3 | mask4))

    return source
