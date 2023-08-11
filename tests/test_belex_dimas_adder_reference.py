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

from typing import Callable, Sequence

import numpy as np

import hypothesis.strategies as st
from hypothesis import given

from open_belex.apl_optimizations import (
    delete_dead_writes, peephole_coalesce_consecutive_and_assignments,
    peephole_coalesce_shift_before_op,
    peephole_coalesce_two_consecutive_sb_from_rl,
    peephole_coalesce_two_consecutive_sb_from_src,
    peephole_eliminate_read_after_write,
    peephole_eliminate_write_read_dependence,
    peephole_merge_rl_from_src_and_rl_from_sb,
    peephole_merge_rl_from_src_and_sb_from_rl, peephole_replace_zero_xor)
from open_belex.bleir.types import Example, ValueParameter
from open_belex.common.constants import NUM_PLATS_PER_APUC
from open_belex.decorators import belex_block
from open_belex.expressions import nibble_and_fold
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

render_optm = lambda method: render_bleir(
        method.__name__+"_opt",
        method,
        optimizations=[
    delete_dead_writes,
    peephole_eliminate_read_after_write,
    peephole_replace_zero_xor,
    peephole_eliminate_write_read_dependence,
    peephole_merge_rl_from_src_and_sb_from_rl,
    peephole_merge_rl_from_src_and_rl_from_sb,
    peephole_coalesce_consecutive_and_assignments,
    peephole_coalesce_two_consecutive_sb_from_src,
    peephole_coalesce_two_consecutive_sb_from_rl,
    peephole_coalesce_shift_before_op,
    ])


u16_strategy = st.integers(min_value=0x0000, max_value=0xFFFF)


def sample_u16s(num_u16s: int) -> Sequence[int]:
    return [u16_strategy.example() for _ in range(num_u16s)]


parameters = [
            ('x', sample_u16s(16)),
            ('y', sample_u16s(16)),
        ]


int2vec = lambda x : [(x >> i) & 0x1 for i in range(16)]

vec2int = lambda v : sum([x*(2**i) for i,x in enumerate(v)])


def dima_nibble_shift4_reference(x,y):
    cout1 = x ^ y
    cout0 = x & y

    cout0 = int2vec(cout0)
    cout1 = int2vec(cout1)

    x = int2vec(x)
    y = int2vec(y)

    for i in range(1,4):
        for j in range(i,16,4):
            cout1[j] = cout1[j-1] & (x[j] ^ y[j])
            cout0[j] = x[j] & y[j] | (cout0[j-1] & (x[j] ^ y[j]))
    carry = 0

    for i in range(0,16,4):
        for j in range(4):
            cout0[i+j] = cout0[i+j] | (cout1[i+j] & carry)

        t = cout0[i+3]
        cout0[i+1], cout0[i+2], cout0[i+3] = cout0[i], cout0[i+1], cout0[i+2]

        cout0[i] = carry
        carry = t

    cout0 = vec2int(cout0)
    x = vec2int(x)
    y = vec2int(y)

    return cout0 ^ y ^ x

@given(x=u16_strategy, y=u16_strategy)
def test_dima_nibble_shift4_reference(x, y):
    expected_value = (x + y) & 0xFFFF
    actual_value = dima_nibble_shift4_reference(x, y)
    assert expected_value == actual_value

@belex_block(build_examples=lambda x : build_expressions_examples_(x, dima_nibble_shift4_reference, parameters))
def dima_belex_nibble_shift4(IR, out, x, y):
    cout0 = IR.var(0)
    cout1 = IR.var(0)
    carry = IR.var(0)

    t = IR.var(0)

    cout1() <= x() ^ y()
    cout0() <= x() & y()

    cout0("159D") <= (cout0("159D")) | (cout1("159D") & cout0("048C"))
    cout0("26AE") <= (cout0("26AE")) | (cout1("26AE") & cout0("159D"))
    cout0("37BF") <= (cout0("37BF")) | (cout1("37BF") & cout0("26AE"))

    # cout1("159D") <= cout1("159D") & cout1("048C")
    # cout1("26AE") <= cout1("26AE") & cout1("159D")
    # cout1("37BF") <= cout1("37BF") & cout1("26AE")

    cout1() <= nibble_and_fold(cout1())

    carry() <= y()^y()  # Workaround: We want to initialize `carry` to 0

    for i in [0, 4, 8, 12]:
       idx = [i, i+1, i+2, i+3]
       cout0(idx) <= (cout0(idx) | (cout1(idx) & carry(idx)))
       t() <= cout0(idx[-1])
       cout0([i+1,i+2,i+3]) <= cout0([i+0,i+1,i+2])
       cout0(i) <= carry(i)
       carry() <= t()

    out() <= (cout0() ^ y() ^ x())
    return out

def test_dima_belex_nibble_shift4():
    # render_bleir("foo-test", dima_belex_nibble_shift4)
    render_optm(dima_belex_nibble_shift4)
