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

from typing import Callable, List, Sequence, Tuple

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
from open_belex_tests.harness import render_bleir


def example_factory(
        value_param_factory: Callable[[str, np.array], ValueParameter],
        expected_value_calculator: Callable[[int, int], int],
        params: List[Tuple[str, int]]) -> Example:

    names = [x[0] for x in params]
    values = [np.repeat(x[1], NUM_PLATS_PER_APUC).astype(np.uint16)
              for x in params]

    expected_value = expected_value_calculator(*values)

    return Example(
        expected_value=value_param_factory('out', expected_value),
        parameters=[value_param_factory(key, val)
                    for key, val in zip(names, values)])


def examples_factory(
        value_param_factory: Callable[[str, np.array], ValueParameter],
        expected_value_calculator: Callable[[int, int], int],
        params: List[Tuple[str, Sequence[int]]]) \
        -> Sequence[Example]:
    num_samples = len(params[0][1])
    result = [
        example_factory(
            value_param_factory,
            expected_value_calculator,
            # example: ('x', 4432)
            [(name, par[i]) for name, par in params])
        for i in range(num_samples) ]
    return result


render_optm = lambda function_under_test: render_bleir(
    function_under_test.__name__ + "_opt",
    function_under_test,
    optimizations=[
        # peephole_eliminate_read_after_write,
        # delete_dead_writes,
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


int2vec = lambda x: [(x >> i) & 0x1 for i in range(16)]

vec2int = lambda v: sum([x*(2**i) for i, x in enumerate(v)])


def dima_nibble_shift4_reference(x: int, y: int) -> int:
    r"""Compute bit-wise nibble-shift sum in pure Python."""
    # compute with 16-bit bit vectors.
    cout0 = int2vec(x & y)
    cout1 = int2vec(x ^ y)
    xv = int2vec(x)
    yv = int2vec(y)

    for i in range(1, 4):
        for j in range(i, 16, 4):
            cout0[j] = xv[j] & yv[j] | (cout0[j - 1] & (xv[j] ^ yv[j]))
            cout1[j] = cout1[j - 1] & (xv[j] ^ yv[j])
    carry = 0

    for j in range(0, 16, 4):
        for i in range(4):
            cout0[i + j] |= (cout1[i + j] & carry)

        t = cout0[j + 3]
        cout0[j + 1], cout0[j + 2], cout0[j + 3] = \
            cout0[j], cout0[j + 1], cout0[j + 2]  # nibble shift!

        cout0[j] = carry
        carry = t

    return vec2int(cout0) ^ y ^ x


@given(x=u16_strategy, y=u16_strategy)
def test_dima_nibble_shift4_reference(x, y):
    expected_value = (x + y) & 0xFFFF
    actual_value = dima_nibble_shift4_reference(x, y)
    assert expected_value == actual_value


# @belex_block(
#     build_examples=lambda x : build_expressions_examples_(
#         x, dima_nibble_shift4_reference, parameters))
# def dima_belex_nibble_shift44444(IR, out, x, y):
#     cout0 = IR.var(0)
#     cout1 = IR.var(0)
#     carry = IR.var(0)

#     xXy = IR.var(0)
#     xy  = IR.var(0)
#     t = IR.var(0)

#     cout1() <= x() ^ y()
#     xXy() <= cout1()
#     cout0() <= x() & y()
#     xy()  <= cout0()

#     # WORKS
#     cout1("159D") <=  cout1("159D") & cout1("048C")
#     cout1("26AE") <=  cout1("26AE") & cout1("159D")
#     cout1("37BF") <=  cout1("37BF") & cout1("26AE")

#     # DOESN'T WORK
#     # cout1("159D") <= cout1("048C") & cout1("159D")
#     # cout1("26AE") <= cout1("159D") & cout1("26AE")
#     # cout1("37BF") <= cout1("26AE") & cout1("37BF")

#     # WORKS
#     # cout1("159D") <= cout1("048C") & xXy("159D")
#     # cout1("26AE") <= cout1("159D") & xXy("26AE")
#     # cout1("37BF") <= cout1("26AE") & xXy("37BF")

#     # WORKS
#     # cout1("159D") <= cout1("048C") & (x("159D") ^ y("159D"))
#     # cout1("26AE") <= cout1("159D") & (x("26AE") ^ y("26AE"))
#     # cout1("37BF") <= cout1("26AE") & (x("37BF") ^ y("37BF"))



#     # DOESN'T WORK
#     # cout0("159D") <= cout0("159D") | cout1("159D") & cout0("048C")
#     # cout0("26AE") <= cout0("26AE") | cout1("26AE") & cout0("159D")
#     # cout0("37BF") <= cout0("37BF") | cout1("37BF") & cout0("26AE")

#     # WORKS
#     cout0("159D") <= xy("159D") | cout0("048C") & xXy("159D")
#     cout0("26AE") <= xy("26AE") | cout0("159D") & xXy("26AE")
#     cout0("37BF") <= xy("37BF") | cout0("26AE") & xXy("37BF")

#     # WORKS
#     # cout0("159D") <= (x("159D") & y("159D")) | (cout0("048C") & (x("159D") ^ y("159D")))
#     # cout0("26AE") <= (x("26AE") & y("26AE")) | (cout0("159D") & (x("26AE") ^ y("26AE")))
#     # cout0("37BF") <= (x("37BF") & y("37BF")) | (cout0("26AE") & (x("37BF") ^ y("37BF")))

#     # DOESN'T WORK
#     # cout0("159D") <= cout0("159D") | (cout0("048C") & cout1("159D"))
#     # cout0("26AE") <= cout0("26AE") | (cout0("159D") & cout1("26AE"))
#     # cout0("37BF") <= cout0("37BF") | (cout0("26AE") & cout1("37BF"))

#     carry() <= y()^y()  # Workaround: We want to initialize `carry` to 0

#     for i in [0, 4, 8, 12]:
#        idx = [i, i+1, i+2, i+3]
#        cout0(idx) <= (cout0(idx) | (cout1(idx) & carry(idx)))
#        t() <= cout0(idx[-1])
#        cout0([i+1,i+2,i+3]) <= cout0([i+0,i+1,i+2])
#        cout0(i) <= carry(i)
#        carry() <= t()

#     out() <= (cout0() ^ y() ^ x())
#     return out


@belex_block(
    build_examples=lambda value_parameter_factory:
        examples_factory(
            value_parameter_factory,
            dima_nibble_shift4_reference,
            parameters))
def dima_belex_nibble_shift5(IR, out, x, y):
    cout0 = IR.var(0)
    cout1 = IR.var(0)
    carry = IR.var(0)

    t = IR.var(0)

    cout1() <= x() ^ y()
    cout0() <= x() & y()

    # RL [:] <= sb6()

    # RL ["0x2222"] |= sb4() & NRL()
    # RL ["0x4444"] |= sb4() & NRL()
    # RL ["0x8888"] |= sb4() & NRL()

    # sb6[:] <= RL()

    # task name: COMPLEX INSTRUCTION (OR-EQUALS)

    cout0("159D") <= cout0("159D") | cout0("048C") & cout1("159D")
    cout0("26AE") <= cout0("26AE") | cout0("159D") & cout1("26AE")
    cout0("37BF") <= cout0("37BF") | cout0("26AE") & cout1("37BF")

    # cout0["159D"] |= cout0("048C") & cout1("159D")
    # cout0["26AE"] |= cout0("159D") & cout1("26AE")
    # cout0["37BF"] |= cout0("26AE") & cout1("37BF")

    # task name : AND-FOLD (follow on: de Morgan's OR-FOLD)
    # cout1 computation becomes es a FOLD

    # # RL ["0xFFFF"] <= sb5()
    # with apl_commands():
    #     RL ["0xAAAA"] &= NRL()
    #     GGL["0x2222"] <= RL()
    # RL ["0xCCCC"] &= GGL()

    cout1("159D") <= cout1("159D") & cout1("048C")
    cout1("26AE") <= cout1("26AE") & cout1("159D")
    cout1("37BF") <= cout1("37BF") & cout1("26AE")

    # task name: SECTION-WISE DEAD WRITES

    carry() <= y() ^ y()  # Workaround: We want to initialize `carry` to 0

    for i in [0, 4, 8, 12]:
       idx = [i, i+1, i+2, i+3]
       cout0(idx) <= (cout0(idx) | (cout1(idx) & carry(idx)))
       t() <= cout0(idx[-1])
       cout0([i+1,i+2,i+3]) <= cout0([i+0,i+1,i+2])
       cout0(i) <= carry(i)
       carry() <= t()

    out() <= (cout0() ^ y() ^ x())
    return out


def test_dima_belex_nibble_shift5():
    render_optm(dima_belex_nibble_shift5)
