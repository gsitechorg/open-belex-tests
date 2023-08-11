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
