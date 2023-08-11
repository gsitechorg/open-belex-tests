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

from typing import Callable, Optional

import numpy as np

from open_belex.apl_optimizations import (delete_dead_writes,
                                          peephole_eliminate_read_after_write)
from open_belex.bleir.types import Example, ValueParameter
from open_belex.common.constants import NUM_PLATS_PER_APUC
from open_belex.decorators import belex_block
from open_belex_tests.harness import render_bleir


def build_carry_cascade_2_example(value_param: Callable[[str, np.array], ValueParameter],
                          A: np.array, B: np.array,
                          expected_value: Optional[np.array] = None) -> Example:
    if expected_value is None:
        sm_048C = 0x1111
        sm_159D = 0x2222
        sm_FFFF = 0xFFFF

        a_FFFF = A
        b_FFFF = B

        a_159D = a_FFFF & sm_159D
        b_159D = b_FFFF & sm_159D
        a_048C = a_FFFF & sm_048C
        b_048C = b_FFFF & sm_048C
        c_159D = (a_159D ^ b_159D) & (((a_048C ^ b_048C) << 1) & sm_FFFF)

        eFFFF = c_159D
        expected_value = eFFFF

    return Example(
        expected_value=value_param('out', expected_value),
        parameters=[value_param('a', A),
                    value_param('b', B)])


def build_carry_cascade_2_examples(value_param):
    seed = 0
    random = np.random.RandomState(seed)

    return [
        build_carry_cascade_2_example(value_param,
            A=np.repeat(0x0110, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0011, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_carry_cascade_2_example(value_param,
            A=np.repeat(0x0111, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0101, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_carry_cascade_2_example(value_param,
            A=np.repeat(0x08EC, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x00B2, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_carry_cascade_2_example(value_param,
            A=np.repeat(0xFFFE, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0001, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_carry_cascade_2_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_carry_cascade_2_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_carry_cascade_2_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
    ]

# addsub_frag_add_u16(RN_REG x, RN_REG y, RN_REG res, RN_REG x_xor_y, RN_REG cout1)
# cout1[0] = X[0]^Y[0];
# cout1[1] = (X[0]^Y[0]) & (X[1]^Y[1]);
# cout1[2] = (X[0]^Y[0]) & (X[1]^Y[1]) & (X[2]^Y[2]);
# cout1[3] = (X[0]^Y[0]) & (X[1]^Y[1]) & (X[2]^Y[2]) & (X[3]^Y[3]);
# cout0[0] = X[0]&Y[0];
# cout0[1] = X[1]&Y[1] | (COUT0[0] & X[1]^Y[1]);
# cout0[2] = X[2]&Y[2] | (COUT0[1] & X[2]^Y[2]);
# cout0[3] = X[3]&Y[3] | (COUT0[2] & X[3]^Y[3]);

@belex_block(build_examples=build_carry_cascade_2_examples)
def carry_cascade_2(IR, out, a, b):
    c = IR.var(0)

    # c("048C") <= a("048C") ^ b("048C")

    # c("159D") <= (a("048C") ^ b("048C")) & (a("159D") ^ b("159D"))

    # c("159D") <= (a("048C") ^ b("048C"))

    c("159D") <= (a("159D") ^ b("159D")) & (a("048C") ^ b("048C"))

    # c("159D") &= (a("048C") ^ b("048C"))

    out() <= out() ^ out()
    out("159D") <= c("159D")

    return out


def test_belex_carry_cascade_2_no_optim():
    render_bleir("carry_cascade_2_no_opt", carry_cascade_2)


def test_belex_carry_cascade_2_with_optim():
    render_bleir("carry_cascade_2_opt", carry_cascade_2, optimizations=[
        peephole_eliminate_read_after_write,
        delete_dead_writes,
    ])


if __name__ == "__main__":
    test_belex_carry_cascade_2_with_optim()
