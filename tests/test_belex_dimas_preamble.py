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
            ('x', [0x1234, 0x5678, 0x9ABC, 0xDEFF, 7, 0x0110, 777, 5443]),
            ('y', [6, 2, 1, 5, 0x0011, 26, 666,12]),
            #('c', [0x2222, 33, 333,999])
        ]



int2vec = lambda x : [(x >> i) & 0x1 for i in range(16)]

vec2int = lambda v : sum([x*(2**i) for i,x in enumerate(v)])

def dima_preamble_cout0(x,y):

    cout0 = int2vec(x & y)

    x = int2vec(x)
    y = int2vec(y)

    for i in range(1,4):
        for j in range(i,16,4):
            cout0[j] = x[j] & y[j] | (cout0[j-1] & (x[j] ^ y[j]))

    cout0 = vec2int(cout0)

    return cout0

def dima_preamble(x,y):

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

@belex_block(build_examples=lambda x : build_expressions_examples_(x, dima_preamble_cout0, parameters))
def dima_belex_preamble_cout0(IR, out, x, y):


    cout0 = IR.var(0)

    cout0() <= x() & y()

    cout0("159D") <= x("159D") & y("159D") | (cout0("048C") & (x("159D") ^ y("159D")))
    cout0("26AE") <= x("26AE") & y("26AE") | (cout0("159D") & (x("26AE") ^ y("26AE")))
    cout0("37BF") <= x("37BF") & y("37BF") | (cout0("26AE") & (x("37BF") ^ y("37BF")))

    out() <= cout0()
    return out

@belex_block(build_examples=lambda x : build_expressions_examples_(x, dima_preamble_cout0, parameters))
def dima_belex_preamble_cout0_constant_propagation1(IR, out, x, y):
    cout0 = IR.var(0)
    temp = IR.var(0)

    cout0() <= x() & y()
    temp() <= cout0() | (x()^y())

    for i in range(1,4):
        for j in range(i,16,4):
            cout0(j) <= (cout0(j) | cout0(j-1))

    out() <= cout0() & temp()

    return out

def test_dima_belex_preamble_cout0():
    render_optm(dima_belex_preamble_cout0)

def test_dima_belex_preamble_cout0_constant_propagation1():
    render_optm(dima_belex_preamble_cout0_constant_propagation1)

@belex_block(build_examples=lambda x : build_expressions_examples_(x, dima_preamble, parameters))
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
