r"""
By Dylon Edwards
"""

from typing import Callable

import numpy as np

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
    # peephole_eliminate_read_after_write,
    # delete_dead_writes,
    # peephole_merge_rl_from_src_and_sb_from_rl,
    # peephole_merge_rl_from_src_and_rl_from_sb,
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


parameters = [
            ('x', [0x1234, 0x5678, 0x9ABC, 0xDEFF, 7, 0x0110, 777, 5443]),
            ('y', [6, 2, 1, 5, 0x0011, 26, 666,12]),
            #('c', [0x2222, 33, 333,999])
        ]


int2vec = lambda x : [(x >> i) & 0x1 for i in range(16)]


vec2int = lambda v : sum([x*(2**i) for i,x in enumerate(v)])


def dima_expr2(x,y):

    cout1 = int2vec(x ^ y)
    cout0 = int2vec(x & y)

    x = int2vec(x)
    y = int2vec(y)

    for i in range(1,4):
        for j in range(i,16,4):
            cout1[j] = cout1[j-1] & (x[j] ^ y[j])
            cout0[j] = x[j] & y[j] | (cout0[j-1] & (x[j] ^ y[j]))

    cout1 = vec2int(cout1) #x ^ y ^ cout0
    cout0 = vec2int(cout0)

    x = vec2int(x)
    y = vec2int(y)

    res = cout0 | cout1

    return res


@belex_block(build_examples=lambda x : build_expressions_examples_(x, dima_expr2, parameters))
def dima_belex_expr2(IR, out, x, y):

    cout1 = IR.var(0)
    cout0 = IR.var(0)
    #carry = IR.var(0)
    #t = IR.var(0)

    cout1() <= x() ^ y()
    cout0() <= x() & y()

    cout1("159D") <= cout1("048C") & (x("159D") ^ y("159D"))
    cout1("26AE") <= cout1("159D") & (x("26AE") ^ y("26AE"))
    cout1("37BF") <= cout1("26AE") & (x("37BF") ^ y("37BF"))

    cout0("159D") <= x("159D") & y("159D") | (cout0("048C") & (x("159D") ^ y("159D")))
    cout0("26AE") <= x("26AE") & y("26AE") | (cout0("159D") & (x("26AE") ^ y("26AE")))
    cout0("37BF") <= x("37BF") & y("37BF") | (cout0("26AE") & (x("37BF") ^ y("37BF")))

    out() <= cout0() | cout1()

    return out


def test_dima_belex_expr2():
    render_optm(dima_belex_expr2)


def dima_nibble_shift(x,y):
    x = int2vec(x)
    for i in range(0,16,4):
        t = x[i+3]
        x[i+1], x[i+2], x[i+3] = x[i], x[i+1], x[i+2]
        x[i] = t

    x = vec2int(x)
    return x


def dima_nibble_shift2(x,y):
    x = int2vec(x)
    carry = 0

    for i in range(0,16,4):
        t = x[i+3]
        x[i+1], x[i+2], x[i+3] = x[i] | carry, x[i+1] | carry, x[i+2] | carry
        x[i] = carry
        carry = t

    x = vec2int(x)
    return x


def dima_nibble_shift4(x,y):

    cout0 = x ^ y
    cout1 = x & y

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


# @belex_block(build_examples=lambda x : build_expressions_examples_(x, dima_nibble_shift4, parameters))
# def dima_belex_nibble_shift4(IR, out, x, y):
#     cout0 = IR.var(0)
#     cout1 = IR.var(0)
#     carry = IR.var(0)

#     t = IR.var(0)

#     cout0() <= x() ^ y()
#     cout1() <= x() & y()

#     cout1("159D") <= cout1("048C") & (x("159D") ^ y("159D"))
#     cout1("26AE") <= cout1("159D") & (x("26AE") ^ y("26AE"))
#     cout1("37BF") <= cout1("26AE") & (x("37BF") ^ y("37BF"))

#     cout0("159D") <= x("159D") & y("159D") | (cout0("048C") & (x("159D") ^ y("159D")))
#     cout0("26AE") <= x("26AE") & y("26AE") | (cout0("159D") & (x("26AE") ^ y("26AE")))
#     cout0("37BF") <= x("37BF") & y("37BF") | (cout0("26AE") & (x("37BF") ^ y("37BF")))

#     carry() <= y()^y()

#     for i in [0, 4, 8, 12]:
#         idx = [i, i+1, i+2, i+3]
#         cout0(idx) <= cout0(idx) | (cout1(idx) & carry(idx))
#         t() <= cout0(idx[-1])
#         cout0([i+1,i+2,i+3]) <= cout0([i+0,i+1,i+2])
#         cout0(i) <= carry(i)
#         carry() <= t()

#     out() <= cout0() ^ y() ^ x()
#     return out

# def test_dima_belex_nibble_shift4():
#     render_optm(dima_belex_nibble_shift4)


def dima_nibble_shift3(x,y):

    cout0 = x ^ y
    cout1 = x & y

    cout0 = int2vec(cout0)
    cout1 = int2vec(cout1)
    carry = 0

    for i in range(0,16,4):
        for j in range(4):
            cout0[i+j] = cout0[i+j] | (cout1[i+j] & carry)

        t = cout0[i+3]
        cout0[i+1], cout0[i+2], cout0[i+3] = cout0[i], cout0[i+1], cout0[i+2]

        cout0[i] = carry
        carry = t

    cout0 = vec2int(cout0)

    return cout0 ^ y ^ x


@belex_block(build_examples=lambda x : build_expressions_examples_(
    x, dima_nibble_shift3, parameters))
def dima_belex_nibble_shift3(IR, out, x, y):
    cout0 = IR.var(0)
    cout1 = IR.var(0)
    carry = IR.var(0)

    t = IR.var(0)

    cout0() <= x() ^ y()
    cout1() <= x() & y()

    carry() <= y()^y()

    for i in [0, 4, 8, 12]:
        idx = [i, i+1, i+2, i+3]
        cout0(idx) <= cout0(idx) | (cout1(idx) & carry(idx))
        t() <= cout0(idx[-1])
        cout0([i+1,i+2,i+3]) <= cout0([i+0,i+1,i+2])
        cout0(i) <= carry(i)
        carry() <= t()

    out() <= cout0() ^ y() ^ x()
    return out


def test_dima_belex_nibble_shift3():
    render_optm(dima_belex_nibble_shift3)


@belex_block(build_examples=lambda x : build_expressions_examples_(x, dima_nibble_shift2, parameters))
def dima_belex_nibble_shift2(IR, out, x, y):
    cout0 = IR.var(0)
    carry = IR.var(0)
    t = IR.var(0)
    carry() <= y()^y()
    cout0() <= x()
    # for i in [0, 8]:
    for i in [0, 4, 8, 12]: # CRASH
        idx = [i, i+1, i+2, i+3]
        t() <= cout0(idx[-1])
        cout0([i+1,i+2,i+3]) <= cout0([i+0,i+1,i+2]) | carry([i,i+1,i+2])
        cout0(i) <= carry(0)
        carry() <= t()

    out() <= cout0()
    return out


def test_dima_belex_nibble_shift2():
    render_optm(dima_belex_nibble_shift2)


def dima_expr3(x,y):

    cout1 = int2vec(x ^ y)
    cout0 = int2vec(x & y)

    x = int2vec(x)
    y = int2vec(y)

    for i in range(1,4):
        for j in range(i,16,4):
            cout1[j] = cout1[j-1] & (x[j] ^ y[j])
            cout0[j] = x[j] & y[j] | (cout0[j-1] & (x[j] ^ y[j]))

    for i in range(0,16,4):
        t = cout0[i+3]
        x[i+1], x[i+2], x[i+3] = x[i], x[i+1], x[i+2]
        x[i] = t

    x = vec2int(x)
    return x


@belex_block(build_examples=lambda x : build_expressions_examples_(x, dima_nibble_shift, parameters))
def dima_belex_nibble_shift(IR, out, x, y):
    cout0 = IR.var(0)
    carry = IR.var(0)
    t = IR.var(0)
    carry() <= y()^y()
    cout0() <= x()
    for i in [0, 4, 8, 12]:
        idx = [i, i+1, i+2, i+3]
        t() <= cout0(idx[-1])
        cout0([i+1,i+2,i+3]) <= cout0([i+0,i+1,i+2]) | carry([i,i+1,i+2])
        cout0(i) <= t(0)

    out() <= cout0()

    return out


def test_dima_belex_nibble_shift():
    render_optm(dima_belex_nibble_shift)


def dima_adder(x,y):
    return x+y


# @belex_block(build_examples=lambda x : build_expressions_examples_(x, dima_adder, parameters))
# def dima_belex_adder(IR, out, x, y):

#     cout1 = IR.var(0)
#     cout0 = IR.var(0)
#     carry = IR.var(0)
#     t = IR.var(0)

#     cout1() <= x() ^ y()
#     cout0() <= x() & y()

#     cout1("159D") <= cout1("048C") & (x("159D") ^ y("159D"))
#     cout1("26AE") <= cout1("159D") & (x("26AE") ^ y("26AE"))
#     cout1("37BF") <= cout1("26AE") & (x("37BF") ^ y("37BF"))

#     cout0("159D") <= x("159D") & y("159D") | (cout0("048C") & (x("159D") ^ y("159D")))
#     cout0("26AE") <= x("26AE") & y("26AE") | (cout0("159D") & (x("26AE") ^ y("26AE")))
#     cout0("37BF") <= x("37BF") & y("37BF") | (cout0("26AE") & (x("37BF") ^ y("37BF")))


#     carry() <= x() ^ x()

#     for i in [0, 4, 8, 12]:
#         idx = [i, i+1, i+2, i+3]
#         cout0(idx) <= cout0(idx) | (cout1(idx) & carry(idx))
#         t() <= cout0(idx[-1])
#         cout0([i+1,i+2,i+3]) <= cout0([i+0,i+1,i+2])
#         cout0(i) <= t(i)
#         carry() <= t()

#     out() <= x() ^ y() ^ cout0() #|cout1()
#     return out

# def test_dima_belex_adder():
#     render_optm(dima_belex_adder)
