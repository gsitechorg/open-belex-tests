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


import pytest

from open_belex.common.constants import (MAX_L1_VALUE, MAX_L2_VALUE,
                                         MAX_RE_VALUE, MAX_RN_VALUE,
                                         MAX_SM_VALUE)
from open_belex.common.seu_layer import SEULayer
from open_belex_tests.utils import seu_context


@seu_context
def test_sm_regs(seu: SEULayer) -> None:
    with pytest.raises(KeyError):
        seu.sm_regs[0]
    seu.sm_regs[0] = 0x0001 << 0
    assert seu.sm_regs[0] == 0x0001 << 0
    seu.sm_regs.SM_REG_0 = MAX_SM_VALUE - (0x0001 << 0)
    assert seu.sm_regs.SM_REG_0 == MAX_SM_VALUE - (0x0001 << 0)
    seu.sm_regs["SM_REG_0"] = 0x0001 << 0
    assert seu.sm_regs["SM_REG_0"] == 0x0001 << 0

    with pytest.raises(KeyError):
        seu.sm_regs[1]
    seu.sm_regs[1] = 0x0001 << 1
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    seu.sm_regs.SM_REG_1 = MAX_SM_VALUE - (0x0001 << 1)
    assert seu.sm_regs.SM_REG_1 == MAX_SM_VALUE - (0x0001 << 1)
    seu.sm_regs["SM_REG_1"] = 0x0001 << 1
    assert seu.sm_regs["SM_REG_1"] == 0x0001 << 1

    with pytest.raises(KeyError):
        seu.sm_regs[2]
    seu.sm_regs[2] = 0x0001 << 2
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    seu.sm_regs.SM_REG_2 = MAX_SM_VALUE - (0x0001 << 2)
    assert seu.sm_regs.SM_REG_2 == MAX_SM_VALUE - (0x0001 << 2)
    seu.sm_regs["SM_REG_2"] = 0x0001 << 2
    assert seu.sm_regs["SM_REG_2"] == 0x0001 << 2

    with pytest.raises(KeyError):
        seu.sm_regs[3]
    seu.sm_regs[3] = 0x0001 << 3
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    seu.sm_regs.SM_REG_3 = MAX_SM_VALUE - (0x0001 << 3)
    assert seu.sm_regs.SM_REG_3 == MAX_SM_VALUE - (0x0001 << 3)
    seu.sm_regs["SM_REG_3"] = 0x0001 << 3
    assert seu.sm_regs["SM_REG_3"] == 0x0001 << 3

    with pytest.raises(KeyError):
        seu.sm_regs[4]
    seu.sm_regs[4] = 0x0001 << 4
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    seu.sm_regs.SM_REG_4 = MAX_SM_VALUE - (0x0001 << 4)
    assert seu.sm_regs.SM_REG_4 == MAX_SM_VALUE - (0x0001 << 4)
    seu.sm_regs["SM_REG_4"] = 0x0001 << 4
    assert seu.sm_regs["SM_REG_4"] == 0x0001 << 4

    with pytest.raises(KeyError):
        seu.sm_regs[5]
    seu.sm_regs[5] = 0x0001 << 5
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    seu.sm_regs.SM_REG_5 = MAX_SM_VALUE - (0x0001 << 5)
    assert seu.sm_regs.SM_REG_5 == MAX_SM_VALUE - (0x0001 << 5)
    seu.sm_regs["SM_REG_5"] = 0x0001 << 5
    assert seu.sm_regs["SM_REG_5"] == 0x0001 << 5

    with pytest.raises(KeyError):
        seu.sm_regs[6]
    seu.sm_regs[6] = 0x0001 << 6
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    seu.sm_regs.SM_REG_6 = MAX_SM_VALUE - (0x0001 << 6)
    assert seu.sm_regs.SM_REG_6 == MAX_SM_VALUE - (0x0001 << 6)
    seu.sm_regs["SM_REG_6"] = 0x0001 << 6
    assert seu.sm_regs["SM_REG_6"] == 0x0001 << 6

    with pytest.raises(KeyError):
        seu.sm_regs[7]
    seu.sm_regs[7] = 0x0001 << 7
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    seu.sm_regs.SM_REG_7 = MAX_SM_VALUE - (0x0001 << 7)
    assert seu.sm_regs.SM_REG_7 == MAX_SM_VALUE - (0x0001 << 7)
    seu.sm_regs["SM_REG_7"] = 0x0001 << 7
    assert seu.sm_regs["SM_REG_7"] == 0x0001 << 7

    with pytest.raises(KeyError):
        seu.sm_regs[8]
    seu.sm_regs[8] = 0x0001 << 8
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    seu.sm_regs.SM_REG_8 = MAX_SM_VALUE - (0x0001 << 8)
    assert seu.sm_regs.SM_REG_8 == MAX_SM_VALUE - (0x0001 << 8)
    seu.sm_regs["SM_REG_8"] = 0x0001 << 8
    assert seu.sm_regs["SM_REG_8"] == 0x0001 << 8

    with pytest.raises(KeyError):
        seu.sm_regs[9]
    seu.sm_regs[9] = 0x0001 << 9
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    assert seu.sm_regs[9] == 0x0001 << 9
    seu.sm_regs.SM_REG_9 = MAX_SM_VALUE - (0x0001 << 9)
    assert seu.sm_regs.SM_REG_9 == MAX_SM_VALUE - (0x0001 << 9)
    seu.sm_regs["SM_REG_9"] = 0x0001 << 9
    assert seu.sm_regs["SM_REG_9"] == 0x0001 << 9

    with pytest.raises(KeyError):
        seu.sm_regs[10]
    seu.sm_regs[10] = 0x0001 << 10
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    assert seu.sm_regs[9] == 0x0001 << 9
    assert seu.sm_regs[10] == 0x0001 << 10
    seu.sm_regs.SM_REG_10 = MAX_SM_VALUE - (0x0001 << 10)
    assert seu.sm_regs.SM_REG_10 == MAX_SM_VALUE - (0x0001 << 10)
    seu.sm_regs["SM_REG_10"] = 0x0001 << 10
    assert seu.sm_regs["SM_REG_10"] == 0x0001 << 10

    with pytest.raises(KeyError):
        seu.sm_regs[11]
    seu.sm_regs[11] = 0x0001 << 11
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    assert seu.sm_regs[9] == 0x0001 << 9
    assert seu.sm_regs[10] == 0x0001 << 10
    assert seu.sm_regs[11] == 0x0001 << 11
    seu.sm_regs.SM_REG_11 = MAX_SM_VALUE - (0x0001 << 11)
    assert seu.sm_regs.SM_REG_11 == MAX_SM_VALUE - (0x0001 << 11)
    seu.sm_regs["SM_REG_11"] = 0x0001 << 11
    assert seu.sm_regs["SM_REG_11"] == 0x0001 << 11

    with pytest.raises(KeyError):
        seu.sm_regs[12]
    seu.sm_regs[12] = 0x0001 << 12
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    assert seu.sm_regs[9] == 0x0001 << 9
    assert seu.sm_regs[10] == 0x0001 << 10
    assert seu.sm_regs[11] == 0x0001 << 11
    assert seu.sm_regs[12] == 0x0001 << 12
    seu.sm_regs.SM_REG_12 = MAX_SM_VALUE - (0x0001 << 12)
    assert seu.sm_regs.SM_REG_12 == MAX_SM_VALUE - (0x0001 << 12)
    seu.sm_regs["SM_REG_12"] = 0x0001 << 12
    assert seu.sm_regs["SM_REG_12"] == 0x0001 << 12

    with pytest.raises(KeyError):
        seu.sm_regs[13]
    seu.sm_regs[13] = 0x0001 << 13
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    assert seu.sm_regs[9] == 0x0001 << 9
    assert seu.sm_regs[10] == 0x0001 << 10
    assert seu.sm_regs[11] == 0x0001 << 11
    assert seu.sm_regs[12] == 0x0001 << 12
    assert seu.sm_regs[13] == 0x0001 << 13
    seu.sm_regs.SM_REG_13 = MAX_SM_VALUE - (0x0001 << 13)
    assert seu.sm_regs.SM_REG_13 == MAX_SM_VALUE - (0x0001 << 13)
    seu.sm_regs["SM_REG_13"] = 0x0001 << 13
    assert seu.sm_regs["SM_REG_13"] == 0x0001 << 13

    with pytest.raises(KeyError):
        seu.sm_regs[14]
    seu.sm_regs[14] = 0x0001 << 14
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    assert seu.sm_regs[9] == 0x0001 << 9
    assert seu.sm_regs[10] == 0x0001 << 10
    assert seu.sm_regs[11] == 0x0001 << 11
    assert seu.sm_regs[12] == 0x0001 << 12
    assert seu.sm_regs[13] == 0x0001 << 13
    assert seu.sm_regs[14] == 0x0001 << 14
    seu.sm_regs.SM_REG_14 = MAX_SM_VALUE - (0x0001 << 14)
    assert seu.sm_regs.SM_REG_14 == MAX_SM_VALUE - (0x0001 << 14)
    seu.sm_regs["SM_REG_14"] = 0x0001 << 14
    assert seu.sm_regs["SM_REG_14"] == 0x0001 << 14

    with pytest.raises(KeyError):
        seu.sm_regs[15]
    seu.sm_regs[15] = 0x0001 << 15
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    assert seu.sm_regs[9] == 0x0001 << 9
    assert seu.sm_regs[10] == 0x0001 << 10
    assert seu.sm_regs[11] == 0x0001 << 11
    assert seu.sm_regs[12] == 0x0001 << 12
    assert seu.sm_regs[13] == 0x0001 << 13
    assert seu.sm_regs[14] == 0x0001 << 14
    assert seu.sm_regs[15] == 0x0001 << 15
    seu.sm_regs.SM_REG_15 = MAX_SM_VALUE - (0x0001 << 15)
    assert seu.sm_regs.SM_REG_15 == MAX_SM_VALUE - (0x0001 << 15)
    seu.sm_regs["SM_REG_15"] = 0x0001 << 15
    assert seu.sm_regs["SM_REG_15"] == 0x0001 << 15


@seu_context
def test_rn_regs(seu: SEULayer) -> None:
    with pytest.raises(KeyError):
        seu.rn_regs[0]
    seu.rn_regs[0] = 0
    assert seu.rn_regs[0] == 0
    seu.rn_regs.RN_REG_0 = MAX_RN_VALUE - 0
    assert seu.rn_regs.RN_REG_0 == MAX_RN_VALUE - 0
    seu.rn_regs["RN_REG_0"] = 0
    assert seu.rn_regs["RN_REG_0"] == 0

    with pytest.raises(KeyError):
        seu.rn_regs[1]
    seu.rn_regs[1] = 1
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    seu.rn_regs.RN_REG_1 = MAX_RN_VALUE - 1
    assert seu.rn_regs.RN_REG_1 == MAX_RN_VALUE - 1
    seu.rn_regs["RN_REG_1"] = 1
    assert seu.rn_regs["RN_REG_1"] == 1

    with pytest.raises(KeyError):
        seu.rn_regs[2]
    seu.rn_regs[2] = 2
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    seu.rn_regs.RN_REG_2 = MAX_RN_VALUE - 2
    assert seu.rn_regs.RN_REG_2 == MAX_RN_VALUE - 2
    seu.rn_regs["RN_REG_2"] = 2
    assert seu.rn_regs["RN_REG_2"] == 2

    with pytest.raises(KeyError):
        seu.rn_regs[3]
    seu.rn_regs[3] = 3
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    seu.rn_regs.RN_REG_3 = MAX_RN_VALUE - 3
    assert seu.rn_regs.RN_REG_3 == MAX_RN_VALUE - 3
    seu.rn_regs["RN_REG_3"] = 3
    assert seu.rn_regs["RN_REG_3"] == 3

    with pytest.raises(KeyError):
        seu.rn_regs[4]
    seu.rn_regs[4] = 4
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    seu.rn_regs.RN_REG_4 = MAX_RN_VALUE - 4
    assert seu.rn_regs.RN_REG_4 == MAX_RN_VALUE - 4
    seu.rn_regs["RN_REG_4"] = 4
    assert seu.rn_regs["RN_REG_4"] == 4

    with pytest.raises(KeyError):
        seu.rn_regs[5]
    seu.rn_regs[5] = 5
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    seu.rn_regs.RN_REG_5 = MAX_RN_VALUE - 5
    assert seu.rn_regs.RN_REG_5 == MAX_RN_VALUE - 5
    seu.rn_regs["RN_REG_5"] = 5
    assert seu.rn_regs["RN_REG_5"] == 5

    with pytest.raises(KeyError):
        seu.rn_regs[6]
    seu.rn_regs[6] = 6
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    seu.rn_regs.RN_REG_6 = MAX_RN_VALUE - 6
    assert seu.rn_regs.RN_REG_6 == MAX_RN_VALUE - 6
    seu.rn_regs["RN_REG_6"] = 6
    assert seu.rn_regs["RN_REG_6"] == 6

    with pytest.raises(KeyError):
        seu.rn_regs[7]
    seu.rn_regs[7] = 7
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    seu.rn_regs.RN_REG_7 = MAX_RN_VALUE - 7
    assert seu.rn_regs.RN_REG_7 == MAX_RN_VALUE - 7
    seu.rn_regs["RN_REG_7"] = 7
    assert seu.rn_regs["RN_REG_7"] == 7

    with pytest.raises(KeyError):
        seu.rn_regs[8]
    seu.rn_regs[8] = 8
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    seu.rn_regs.RN_REG_8 = MAX_RN_VALUE - 8
    assert seu.rn_regs.RN_REG_8 == MAX_RN_VALUE - 8
    seu.rn_regs["RN_REG_8"] = 8
    assert seu.rn_regs["RN_REG_8"] == 8

    with pytest.raises(KeyError):
        seu.rn_regs[9]
    seu.rn_regs[9] = 9
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    assert seu.rn_regs[9] == 9
    seu.rn_regs.RN_REG_9 = MAX_RN_VALUE - 9
    assert seu.rn_regs.RN_REG_9 == MAX_RN_VALUE - 9
    seu.rn_regs["RN_REG_9"] = 9
    assert seu.rn_regs["RN_REG_9"] == 9

    with pytest.raises(KeyError):
        seu.rn_regs[10]
    seu.rn_regs[10] = 10
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    assert seu.rn_regs[9] == 9
    assert seu.rn_regs[10] == 10
    seu.rn_regs.RN_REG_10 = MAX_RN_VALUE - 10
    assert seu.rn_regs.RN_REG_10 == MAX_RN_VALUE - 10
    seu.rn_regs["RN_REG_10"] = 10
    assert seu.rn_regs["RN_REG_10"] == 10

    with pytest.raises(KeyError):
        seu.rn_regs[11]
    seu.rn_regs[11] = 11
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    assert seu.rn_regs[9] == 9
    assert seu.rn_regs[10] == 10
    assert seu.rn_regs[11] == 11
    seu.rn_regs.RN_REG_11 = MAX_RN_VALUE - 11
    assert seu.rn_regs.RN_REG_11 == MAX_RN_VALUE - 11
    seu.rn_regs["RN_REG_11"] = 11
    assert seu.rn_regs["RN_REG_11"] == 11

    with pytest.raises(KeyError):
        seu.rn_regs[12]
    seu.rn_regs[12] = 12
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    assert seu.rn_regs[9] == 9
    assert seu.rn_regs[10] == 10
    assert seu.rn_regs[11] == 11
    assert seu.rn_regs[12] == 12
    seu.rn_regs.RN_REG_12 = MAX_RN_VALUE - 12
    assert seu.rn_regs.RN_REG_12 == MAX_RN_VALUE - 12
    seu.rn_regs["RN_REG_12"] = 12
    assert seu.rn_regs["RN_REG_12"] == 12

    with pytest.raises(KeyError):
        seu.rn_regs[13]
    seu.rn_regs[13] = 13
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    assert seu.rn_regs[9] == 9
    assert seu.rn_regs[10] == 10
    assert seu.rn_regs[11] == 11
    assert seu.rn_regs[12] == 12
    assert seu.rn_regs[13] == 13
    seu.rn_regs.RN_REG_13 = MAX_RN_VALUE - 13
    assert seu.rn_regs.RN_REG_13 == MAX_RN_VALUE - 13
    seu.rn_regs["RN_REG_13"] = 13
    assert seu.rn_regs["RN_REG_13"] == 13

    with pytest.raises(KeyError):
        seu.rn_regs[14]
    seu.rn_regs[14] = 14
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    assert seu.rn_regs[9] == 9
    assert seu.rn_regs[10] == 10
    assert seu.rn_regs[11] == 11
    assert seu.rn_regs[12] == 12
    assert seu.rn_regs[13] == 13
    assert seu.rn_regs[14] == 14
    seu.rn_regs.RN_REG_14 = MAX_RN_VALUE - 14
    assert seu.rn_regs.RN_REG_14 == MAX_RN_VALUE - 14
    seu.rn_regs["RN_REG_14"] = 14
    assert seu.rn_regs["RN_REG_14"] == 14

    with pytest.raises(KeyError):
        seu.rn_regs[15]
    seu.rn_regs[15] = 15
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    assert seu.rn_regs[9] == 9
    assert seu.rn_regs[10] == 10
    assert seu.rn_regs[11] == 11
    assert seu.rn_regs[12] == 12
    assert seu.rn_regs[13] == 13
    assert seu.rn_regs[14] == 14
    assert seu.rn_regs[15] == 15
    seu.rn_regs.RN_REG_15 = MAX_RN_VALUE - 15
    assert seu.rn_regs.RN_REG_15 == MAX_RN_VALUE - 15
    seu.rn_regs["RN_REG_15"] = 15
    assert seu.rn_regs["RN_REG_15"] == 15


@seu_context
def test_re_regs(seu: SEULayer) -> None:
    with pytest.raises(KeyError):
        seu.re_regs[0]
    seu.re_regs[0] = 0x00FFFF << 0
    assert seu.re_regs[0] == 0x00FFFF << 0
    seu.re_regs.RE_REG_0 = MAX_RE_VALUE - (0x00FFFF << 0)
    assert seu.re_regs.RE_REG_0 == MAX_RE_VALUE - (0x00FFFF << 0)
    seu.re_regs["RE_REG_0"] = 0x00FFFF << 0
    assert seu.re_regs["RE_REG_0"] == 0x00FFFF << 0

    with pytest.raises(KeyError):
        seu.re_regs[1]
    seu.re_regs[1] = 0x00FFFF << 1
    assert seu.re_regs[0] == 0x00FFFF << 0
    assert seu.re_regs[1] == 0x00FFFF << 1
    seu.re_regs.RE_REG_1 = MAX_RE_VALUE - (0x00FFFF << 1)
    assert seu.re_regs.RE_REG_1 == MAX_RE_VALUE - (0x00FFFF << 1)
    seu.re_regs["RE_REG_1"] = 0x00FFFF << 1
    assert seu.re_regs["RE_REG_1"] == 0x00FFFF << 1

    with pytest.raises(KeyError):
        seu.re_regs[2]
    seu.re_regs[2] = 0x00FFFF << 2
    assert seu.re_regs[0] == 0x00FFFF << 0
    assert seu.re_regs[1] == 0x00FFFF << 1
    assert seu.re_regs[2] == 0x00FFFF << 2
    seu.re_regs.RE_REG_2 = MAX_RE_VALUE - (0x00FFFF << 2)
    assert seu.re_regs.RE_REG_2 == MAX_RE_VALUE - (0x00FFFF << 2)
    seu.re_regs["RE_REG_2"] = 0x00FFFF << 2
    assert seu.re_regs["RE_REG_2"] == 0x00FFFF << 2

    with pytest.raises(KeyError):
        seu.re_regs[3]
    seu.re_regs[3] = 0x00FFFF << 3
    assert seu.re_regs[0] == 0x00FFFF << 0
    assert seu.re_regs[1] == 0x00FFFF << 1
    assert seu.re_regs[2] == 0x00FFFF << 2
    assert seu.re_regs[3] == 0x00FFFF << 3
    seu.re_regs.RE_REG_3 = MAX_RE_VALUE - (0x00FFFF << 3)
    assert seu.re_regs.RE_REG_3 == MAX_RE_VALUE - (0x00FFFF << 3)
    seu.re_regs["RE_REG_3"] = 0x00FFFF << 3
    assert seu.re_regs["RE_REG_3"] == 0x00FFFF << 3


@seu_context
def test_ewe_regs(seu: SEULayer) -> None:
    with pytest.raises(KeyError):
        seu.ewe_regs[0]
    seu.ewe_regs[0] = 0x0FF
    assert seu.ewe_regs[0] == 0x0FF
    seu.ewe_regs.EWE_REG_0 = 0x0FF - 0xFF
    assert seu.ewe_regs.EWE_REG_0 == (0, 0xFF - 0xFF)
    seu.ewe_regs.EWE_REG_0 = (0, 0xFF)
    assert seu.ewe_regs.EWE_REG_0 == (0, 0xFF)
    seu.ewe_regs["EWE_REG_0"] = 0x0FF - 0xFF
    assert seu.ewe_regs["EWE_REG_0"] == (0, 0xFF - 0xFF)
    seu.ewe_regs["EWE_REG_0"] = (0, 0xFF)
    assert seu.ewe_regs["EWE_REG_0"] == (0, 0xFF)

    with pytest.raises(KeyError):
        seu.ewe_regs[1]
    seu.ewe_regs[1] = 0x10F
    assert seu.ewe_regs[0] == 0x0FF
    assert seu.ewe_regs[1] == 0x10F
    seu.ewe_regs.EWE_REG_1 = 0x1FF - 0x0F
    assert seu.ewe_regs.EWE_REG_1 == (1, 0xFF - 0x0F)
    seu.ewe_regs.EWE_REG_1 = (1, 0x0F)
    assert seu.ewe_regs.EWE_REG_1 == (1, 0x0F)
    seu.ewe_regs["EWE_REG_1"] = 0x1FF - 0x0F
    assert seu.ewe_regs["EWE_REG_1"] == (1, 0xFF - 0x0F)
    seu.ewe_regs["EWE_REG_1"] = (1, 0x0F)
    assert seu.ewe_regs["EWE_REG_1"] == (1, 0x0F)

    with pytest.raises(KeyError):
        seu.ewe_regs[2]
    seu.ewe_regs[2] = 0x234
    assert seu.ewe_regs[0] == 0x0FF
    assert seu.ewe_regs[1] == 0x10F
    assert seu.ewe_regs[2] == 0x234
    seu.ewe_regs.EWE_REG_2 = 0x2FF - 0x34
    assert seu.ewe_regs.EWE_REG_2 == (2, 0xFF - 0x34)
    seu.ewe_regs.EWE_REG_2 = (2, 0x34)
    assert seu.ewe_regs.EWE_REG_2 == (2, 0x34)
    seu.ewe_regs["EWE_REG_2"] = 0x2FF - 0x34
    assert seu.ewe_regs["EWE_REG_2"] == (2, 0xFF - 0x34)
    seu.ewe_regs["EWE_REG_2"] = (2, 0x34)
    assert seu.ewe_regs["EWE_REG_2"] == (2, 0x34)

    with pytest.raises(KeyError):
        seu.ewe_regs[3]
    seu.ewe_regs[3] = 0x2FF
    assert seu.ewe_regs[0] == 0x0FF
    assert seu.ewe_regs[1] == 0x10F
    assert seu.ewe_regs[2] == 0x234
    assert seu.ewe_regs[3] == 0x2FF
    seu.ewe_regs.EWE_REG_3 = 0x2FF - 0xFF
    assert seu.ewe_regs.EWE_REG_3 == (2, 0xFF - 0xFF)
    seu.ewe_regs.EWE_REG_3 = (2, 0xFF)
    assert seu.ewe_regs.EWE_REG_3 == (2, 0xFF)
    seu.ewe_regs["EWE_REG_3"] = 0x2FF - 0xFF
    assert seu.ewe_regs["EWE_REG_3"] == (2, 0xFF - 0xFF)
    seu.ewe_regs["EWE_REG_3"] = (2, 0xFF)
    assert seu.ewe_regs["EWE_REG_3"] == (2, 0xFF)


@seu_context
def test_l1_regs(seu: SEULayer) -> None:
    with pytest.raises(KeyError):
        seu.l1_regs[0]
    seu.l1_regs[0] = 0
    assert seu.l1_regs[0] == 0
    seu.l1_regs.L1_ADDR_REG_0 = MAX_L1_VALUE
    assert seu.l1_regs.L1_ADDR_REG_0 == (3, 3, 255)
    seu.l1_regs.L1_ADDR_REG_0 = (0,)
    assert seu.l1_regs.L1_ADDR_REG_0 == (0, 0, 0)
    seu.l1_regs.L1_ADDR_REG_0 = (1, 255)
    assert seu.l1_regs.L1_ADDR_REG_0 == (0, 1, 255)
    seu.l1_regs.L1_ADDR_REG_0 = (2, 2, 0)
    assert seu.l1_regs.L1_ADDR_REG_0 == (2, 2, 0)
    seu.l1_regs["L1_ADDR_REG_0"] = MAX_L1_VALUE
    assert seu.l1_regs["L1_ADDR_REG_0"] == (3, 3, 255)
    seu.l1_regs["L1_ADDR_REG_0"] = (0,)
    assert seu.l1_regs["L1_ADDR_REG_0"] == (0, 0, 0)
    seu.l1_regs["L1_ADDR_REG_0"] = (1, 255)
    assert seu.l1_regs["L1_ADDR_REG_0"] == (0, 1, 255)
    seu.l1_regs["L1_ADDR_REG_0"] = (3, 1, 0)
    assert seu.l1_regs["L1_ADDR_REG_0"] == (3, 1, 0)
    seu.l1_regs[0] = 0

    with pytest.raises(KeyError):
        seu.l1_regs[1]
    seu.l1_regs[1] = 16
    assert seu.l1_regs[0] == 0
    assert seu.l1_regs[1] == 16
    seu.l1_regs.L1_ADDR_REG_1 = MAX_L1_VALUE - 16
    assert seu.l1_regs.L1_ADDR_REG_1 == (3, 3, 239)
    seu.l1_regs.L1_ADDR_REG_1 = (16,)
    assert seu.l1_regs.L1_ADDR_REG_1 == (0, 0, 16)
    seu.l1_regs.L1_ADDR_REG_1 = (1, 239)
    assert seu.l1_regs.L1_ADDR_REG_1 == (0, 1, 239)
    seu.l1_regs.L1_ADDR_REG_1 = (2, 1, 16)
    assert seu.l1_regs.L1_ADDR_REG_1 == (2, 1, 16)
    seu.l1_regs["L1_ADDR_REG_1"] = 16
    assert seu.l1_regs["L1_ADDR_REG_1"] == (0, 0, 16)
    seu.l1_regs["L1_ADDR_REG_1"] = (239,)
    assert seu.l1_regs["L1_ADDR_REG_1"] == (0, 0, 239)
    seu.l1_regs["L1_ADDR_REG_1"] = (1, 16,)
    assert seu.l1_regs["L1_ADDR_REG_1"] == (0, 1, 16)
    seu.l1_regs["L1_ADDR_REG_1"] = (1, 2, 239,)
    assert seu.l1_regs["L1_ADDR_REG_1"] == (1, 2, 239)
    seu.l1_regs[1] = 16

    with pytest.raises(KeyError):
        seu.l1_regs[2]
    seu.l1_regs[2] = 32
    assert seu.l1_regs[0] == 0
    assert seu.l1_regs[1] == 16
    assert seu.l1_regs[2] == 32
    seu.l1_regs.L1_ADDR_REG_2 = MAX_L1_VALUE - 32
    assert seu.l1_regs.L1_ADDR_REG_2 == (3, 3, 223)
    seu.l1_regs.L1_ADDR_REG_2 = (32,)
    assert seu.l1_regs.L1_ADDR_REG_2 == (0, 0, 32)
    seu.l1_regs.L1_ADDR_REG_2 = (1, 223)
    assert seu.l1_regs.L1_ADDR_REG_2 == (0, 1, 223)
    seu.l1_regs.L1_ADDR_REG_2 = (2, 1, 32)
    assert seu.l1_regs.L1_ADDR_REG_2 == (2, 1, 32)
    seu.l1_regs["L1_ADDR_REG_2"] = 32
    assert seu.l1_regs["L1_ADDR_REG_2"] == (0, 0, 32)
    seu.l1_regs["L1_ADDR_REG_2"] = (223,)
    assert seu.l1_regs["L1_ADDR_REG_2"] == (0, 0, 223)
    seu.l1_regs["L1_ADDR_REG_2"] = (1, 32,)
    assert seu.l1_regs["L1_ADDR_REG_2"] == (0, 1, 32)
    seu.l1_regs["L1_ADDR_REG_2"] = (1, 2, 223,)
    assert seu.l1_regs["L1_ADDR_REG_2"] == (1, 2, 223)
    seu.l1_regs[2] = 32

    with pytest.raises(KeyError):
        seu.l1_regs[3]
    seu.l1_regs[3] = 48
    assert seu.l1_regs[0] == 0
    assert seu.l1_regs[1] == 16
    assert seu.l1_regs[2] == 32
    assert seu.l1_regs[3] == 48
    seu.l1_regs.L1_ADDR_REG_3 = MAX_L1_VALUE - 48
    assert seu.l1_regs.L1_ADDR_REG_3 == (3, 3, 207)
    seu.l1_regs.L1_ADDR_REG_3 = (48,)
    assert seu.l1_regs.L1_ADDR_REG_3 == (0, 0, 48)
    seu.l1_regs.L1_ADDR_REG_3 = (1, 207)
    assert seu.l1_regs.L1_ADDR_REG_3 == (0, 1, 207)
    seu.l1_regs.L1_ADDR_REG_3 = (2, 1, 48)
    assert seu.l1_regs.L1_ADDR_REG_3 == (2, 1, 48)
    seu.l1_regs["L1_ADDR_REG_3"] = 48
    assert seu.l1_regs["L1_ADDR_REG_3"] == (0, 0, 48)
    seu.l1_regs["L1_ADDR_REG_3"] = (207,)
    assert seu.l1_regs["L1_ADDR_REG_3"] == (0, 0, 207)
    seu.l1_regs["L1_ADDR_REG_3"] = (1, 48,)
    assert seu.l1_regs["L1_ADDR_REG_3"] == (0, 1, 48)
    seu.l1_regs["L1_ADDR_REG_3"] = (1, 2, 207,)
    assert seu.l1_regs["L1_ADDR_REG_3"] == (1, 2, 207)
    seu.l1_regs[3] = 48


@seu_context
def test_l2_regs(seu: SEULayer) -> None:
    with pytest.raises(KeyError):
        seu.l2_regs[0]
    seu.l2_regs[0] = 42
    assert seu.l2_regs[0] == 42
    seu.l2_regs.L2_ADDR_REG_0 = MAX_L2_VALUE - 42
    assert seu.l2_regs.L2_ADDR_REG_0 == MAX_L2_VALUE - 42
    seu.l2_regs["L2_ADDR_REG_0"] = 42
    assert seu.l2_regs["L2_ADDR_REG_0"] == 42
