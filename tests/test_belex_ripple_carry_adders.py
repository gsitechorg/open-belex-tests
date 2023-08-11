r"""By Dylon Edwards and Brian Beckman

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

from open_belex.common.constants import NSECTIONS
from open_belex.literal import (GL, NRL, RL, RSP16, SM_0X0001, SM_0XFFFF, VR,
                                apl_commands, belex_apl)
from open_belex.utils.config_utils import belex_config

from open_belex_tests.utils import belex_property_test

#  _   _      ___       _             _    _       _  __    ___ _ _  __
# | | | |_ _ / _ \ _ __| |_   __ _ __| |__| | _  _/ |/ /   / / | | |_\ \
# | |_| | ' \ (_) | '_ \  _| / _` / _` / _` || || | / _ \ | || | | '_ \ |
#  \___/|_||_\___/| .__/\__| \__,_\__,_\__,_|_\_,_|_\___/ | ||_|_|_.__/ |
#                 |_|                      |___|           \_\       /_/

NUM_BITS = NSECTIONS

@belex_apl
def add_u16_w_ripple_carry(BELEX, out: VR, a: VR, b: VR):
    """Add 2 x 16-bit registers that hold unsigned short values. Add 2 bits from
    section 0, write the 'sum' and propagate the carry to section 1. Then we add
    2 bits from section 1 and also the previous carry, write the 'sum' and
    propagate the carry to section 2, etc.

    Example:
                                0101  (a)
                              + 0011  (b)
                              ------
                                1000  (sum)

    Mentally rotate the example 90 degrees counterclockwise. See the least
    significant bits of each addend in section 0, etc.

                                0101  (a) ~~~> 1
                                               0
                                               1
                                               0

    and likewise for b. Now we can see the computation take place in the
    half-bank:

    1st iteration:
                a b  cry next_cry  sum
    section#0:  1 1  0    1        0      <- a + b + cry
    section#1:  0 1  0
    section#2:  1 0  0
    section#3:  0 0  0

    2nd iteration:
                a b  cry next_cry  sum
    section#0:  1 1  0    1        0
    section#1:  0 1  1    1        0      <- (cry = next_cry from
    section#2:  1 0  0                         previous iteration)
    section#3:  0 0  0

    3rd iteration:
                a b  cry next_cry  sum
    section#0:  1 1  0    1        0
    section#1:  0 1  1    1        0
    section#2:  1 0  1    1        0      <- (cry = next_cry from
    section#3:  0 0  0                         previous iteration)

    4th iteration:
                a b  cry next_cry  sum
    section#0:  1 1  0    1        0
    section#1:  0 1  1    1        0
    section#2:  1 0  1    1        0
    section#3:  0 0  1    0        1      <- (cry = next_cry from
                                               previous iteration)
    sum = a ^ b ^ cry
    cry = a&b | a&cry | b&cry"""

    cry = BELEX.VR()

    # Temporary registers (for cry = a&b | a&cry | b&cry)
    a_and_b = BELEX.VR()
    a_and_cry = BELEX.VR()
    b_and_cry = BELEX.VR()

    RL[0:NUM_BITS] <= a()
    RL[0:NUM_BITS] ^= b()
    out[0:NUM_BITS] <= RL()

    RL[0:NUM_BITS] <= a()
    RL[0:NUM_BITS] &= b()
    a_and_b[0:NUM_BITS] <= RL()

    cry[1] <= NRL()

    for sec in range(1, NUM_BITS - 1):
        RL[sec] <= out()
        RL[sec] ^= cry()
        out[sec] <= RL()

        RL[sec] <= a()
        RL[sec] &= cry()
        a_and_cry[sec] <= RL()

        RL[sec] <= b()
        RL[sec] &= cry()
        b_and_cry[sec] <= RL()

        RL[sec] <= a_and_b()
        RL[sec] |= a_and_cry()
        RL[sec] |= b_and_cry()
        cry[sec + 1] <= NRL()

    RL[NUM_BITS - 1] <= cry()
    RL[NUM_BITS - 1] ^= out()
    out[NUM_BITS - 1] <= RL()


@belex_property_test(add_u16_w_ripple_carry)
def test_add_u16_w_ripple_carry(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) & (2 ** NUM_BITS - 1)


# This is a long-running test so just run it once
@belex_config(max_rn_regs=4)
@belex_property_test(add_u16_w_ripple_carry, max_examples=1)
def test_add_u16_w_carry_pred_w_4_rn_regs(
        a: np.ndarray,
        b: np.ndarray) -> np.ndarray:
    return (a + b) & (2 ** NUM_BITS - 1)


#    ___       _             _    _       _  __
#   / _ \ _ __| |_   __ _ __| |__| | _  _/ |/ /
#  | (_) | '_ \  _| / _` / _` / _` || || | / _ \
#   \___/| .__/\__| \__,_\__,_\__,_|_\_,_|_\___/
#     ___|_|              _       |___|_ _   _          __
#    / / |_  __ _ _ _  __| |_ __ ___ _(_) |_| |_ ___ _ _\ \
#   | || ' \/ _` | ' \/ _` \ V  V / '_| |  _|  _/ -_) ' \| |
#   | ||_||_\__,_|_||_\__,_|\_/\_/|_| |_|\__|\__\___|_||_| |
#    \_\                                                /_/


@belex_apl
def add_u16_24_nov(Belex, out: VR, a: VR, b: VR):
    t = Belex.VR()
    os = SM_0X0001
    fs = SM_0XFFFF

    t[os] <= RSP16()
    RL[fs] <= a() & b()
    GL[os] <= RL()

    for sh in range(1, 15):
        t[os << sh] <= GL()
        RL[os << sh] |= a() & GL()
        RL[os << sh] |= b() & GL()
        GL[os << sh] <= RL()

    sh = 15
    t[os << sh] <= GL()

    RL[fs] <= a()
    RL[fs] ^= b()
    RL[fs] ^= t()
    out[fs] <= RL()


@belex_property_test(add_u16_24_nov)
def test_add_u16_24_nov(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


#   ___                  ___  __           _ _   _      _  _ ___ _
#  |_ _|_______  _ ___  | __|/  \  __ __ _(_) |_| |_   | \| | _ \ |
#   | |(_-<_-< || / -_) |__ \ () | \ V  V / |  _| ' \  | .` |   / |__
#  |___/__/__/\_,_\___| |___/\__/   \_/\_/|_|\__|_||_| |_|\_|_|_\____|


@belex_apl
def add_u16_issue_50(Belex, out: VR, a: VR, b: VR):
    t = Belex.VR()
    os = SM_0X0001
    fs = SM_0XFFFF

    with apl_commands("instruction 1"):
        t[os] <= RSP16()
        RL[fs] <= a() & b()

    for sh in range(1, 15):
        with apl_commands(f"instruction {sh * 2 + 0}"):
            t[os << sh] <= NRL()
            RL[os << sh] |= a() & NRL()
        with apl_commands(f"instruction {sh * 2 + 1}"):
            RL[os << sh] |= b() & NRL()

    sh = 15

    with apl_commands("instruction 30"):
        t[os << sh] <= NRL()
        RL[fs] <= a()
    with apl_commands("instruction 31"):
        RL[fs] ^= b()
    with apl_commands("instruction 32"):
        RL[fs] ^= t()
    with apl_commands("instruction 33"):
        out[fs] <= RL()


@belex_property_test(add_u16_issue_50)
def test_add_u16_issue_50(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


#   ____   ___           ___ __ ___ __            _    _       _  __
#  |__  | |   \ ___ __  |_  )  \_  )  \   __ _ __| |__| | _  _/ |/ /
#    / /  | |) / -_) _|  / / () / / () | / _` / _` / _` || || | / _ \
#   /_/   |___/\___\__| /___\__/___\__/  \__,_\__,_\__,_|_\_,_|_\___/
#                                                      |___|


@belex_apl
def add_u16_7_dec_2020(Belex, out: VR, a: VR, b: VR):
    cry = Belex.VR()

    a_xor_b = Belex.VR()

    # Temporary registers (for cry = a&b | a&cry | b&cry)
    a_and_b = Belex.VR()
    a_and_cry = Belex.VR()
    b_and_cry = Belex.VR()
    a_and_b_or_a_and_cry = Belex.VR()

    RL["0xFFFF"] <= a()
    RL["0xFFFF"] ^= b()
    a_xor_b["0xFFFF"] <= RL()

    RL["0xFFFF"] <= a()
    RL["0xFFFF"] &= b()
    a_and_b["0xFFFE"] <= RL()
    cry["0x0001"] <= RL()

    for sec in range(1, 16):
        cry[sec] <= NRL()

        RL[sec] <= a()
        RL[sec] &= cry()
        a_and_cry[sec] <= RL()

        RL[sec] <= b()
        RL[sec] &= cry()
        b_and_cry[sec] <= RL()

        RL[sec] <= a_and_b()
        RL[sec] |= a_and_cry()
        a_and_b_or_a_and_cry[sec] <= RL()

        RL[sec] <= a_and_b_or_a_and_cry()
        RL[sec] |= b_and_cry()
        cry[sec] <= RL()

    RL["0xFFFF"] <= cry()
    cry["0xFFFF"] <= NRL()

    RL["0xFFFF"] <= a_xor_b()
    RL["0xFFFF"] ^= cry()
    out["0xFFFF"] <= RL()


@belex_property_test(add_u16_7_dec_2020)
def test_add_u16_7_dec_2020(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


@belex_config(max_rn_regs=5)
@belex_property_test(add_u16_7_dec_2020)
def test_add_u16_7_dec_2020_w_5_rn_regs(
        a: np.ndarray,
        b: np.ndarray) -> np.ndarray:
    return (a + b) & (2 ** NUM_BITS - 1)
