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

from typing import Callable

import numpy as np

from open_belex.apl_optimizations import (delete_dead_writes,
                                          peephole_eliminate_read_after_write)
from open_belex.bleir.types import Example, ValueParameter
from open_belex.common.constants import NUM_PLATS_PER_APUC
from open_belex.decorators import belex_block
from open_belex_tests.harness import render_bleir


def build_expressions_example1 (value_param: Callable[[str, np.array], ValueParameter],
        params:dict, f) -> Example:

    names = [x[0] for x in params]
    values = [np.repeat(x[1], NUM_PLATS_PER_APUC).astype(np.uint16) for x in params]

    expected_value = f(*values)

    return Example(
        expected_value=value_param('out', expected_value),
        parameters=[value_param(key, val) for key, val in zip(names, values)])


def build_expressions_examples_(value_param, f, params):
    return [build_expressions_example1(value_param, [(name, par[i]) for name,par in parameters], f)
            for i in range(len(parameters[0][1]))]

render_optm = lambda method: render_bleir(method.__name__+"_opt", method, optimizations=[
    peephole_eliminate_read_after_write,
    delete_dead_writes,
    ])


parameters = [
            ('a', [0x0110, 25, 777, 5443]),
            ('b', [0x0011, 26, 666,12]),
            ('c', [0x2222, 33, 333,999])
        ]

f = lambda a,b,c : (a^b) |(a^c)|(b^c)


@belex_block(build_examples=lambda x : build_expressions_examples_(x, f, parameters))
def expressions6(IR, out, a, b, c):
    t1 = IR.var(0)
    t2 = IR.var(0)
    t3 = IR.var(0)

    t1() <= a() ^ b()
    t2() <= t1() | (a() ^ c())
    t3() <= t2() | (b() ^ c())

    out() <= t3()

    return out

def test_6():
    render_optm(expressions6)

if __name__ == "__main__":
    test_belex_expressions_with_optim()
