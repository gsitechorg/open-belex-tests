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

NUM_BITS = 16


idx = False
def build_carry_cascade_6_example(value_param: Callable[[str, np.array], ValueParameter],
                                  A: np.array, B: np.array, C: np.array,
                                  expected_value: Optional[np.array] = None) -> Example:
    global NUM_BITS
    if expected_value is None:
        x159D = A & 0x2222
        y048C = B & 0x1111
        e159D = (x159D ^ (y048C << 1))
        expected_value = (C & ~0x2222) | e159D

    return Example(
        expected_value=value_param('out', expected_value),
        parameters=[value_param('a', A),
                    value_param('b', B),
                    value_param('c', C)])


def build_carry_cascade_6_examples(value_param):
    seed = 0
    random = np.random.RandomState(seed)

    return [
        build_carry_cascade_6_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            C=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_carry_cascade_6_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            C=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_carry_cascade_6_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            C=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
    ]

@belex_block(build_examples=build_carry_cascade_6_examples)
def carry_cascade_6(IR, out, a, b, c):
    # c = IR.var(0)

    # c("048C") <= a("048C") ^ b("048C")

    # c("159D") <= (a("048C") ^ b("048C")) & (a("159D") ^ b("159D"))

    out() <= c()
    out("159D") <= (a("159D") ^ b("048C"))
#   c("159D") <= (b("048C") ^ a("159D"))
#   c("048C") <= (a("159D") ^ b("048C"))

    # c("159D") <= (a("159D") ^ b("159D")) & (a("048C") ^ b("048C"))

    # out() <= c()
    # out("159D") <= c("159D")

    return out


def test_belex_carry_cascade_6_no_optim():
    render_bleir("carry_cascade_6_no_opt", carry_cascade_6)


def test_belex_carry_cascade_6_with_optim():
    render_bleir("carry_cascade_6_opt", carry_cascade_6, optimizations=[
        peephole_eliminate_read_after_write,
        delete_dead_writes,
    ])


if __name__ == "__main__":
    test_belex_carry_cascade_6_with_optim()
