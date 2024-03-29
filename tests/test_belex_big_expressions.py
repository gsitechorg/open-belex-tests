r"""
By Dylon Edwards
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

f = lambda a,b,c : (a^b) & c

@belex_block(build_examples=lambda x : build_expressions_examples_(x, f, parameters))
def expressions(IR, out, a, b, c):
    d = IR.var(0)

    d() <= a() ^ b()

    out() <= d() & c()

    return out

def test_0():
    render_optm(expressions)

@belex_block(build_examples=lambda x : build_expressions_examples_(x, f, parameters))
def expressions1(IR, out, a, b, c):

    out() <= (a() ^ b()) & c()

    return out

def test_1():
    render_optm(expressions1)

@belex_block(build_examples=lambda x : build_expressions_examples_(x, f, parameters))
def expressions1a(IR, out, a, b, c):

    for i in range(16):
        out(i) <= (a(i) ^ b(i)) & c(i)

    return out

def test_1a():
    render_optm(expressions1a)

g = lambda a,b,c : (a^b)|(b^c)|(a^c)


@belex_block(build_examples=lambda x : build_expressions_examples_(x, g, parameters))
def expressions2(IR, out, a, b, c):

    t1 = IR.var(0)
    t2 = IR.var(0)
    t3 = IR.var(0)

    t1() <= a() ^ b()
    t2() <= a() ^ c()
    t3() <= b() ^ c()

    out() <= t1() | t2() | t3()
    return out

def test_2():
    render_optm(expressions2)

@belex_block(build_examples=lambda x : build_expressions_examples_(x, g, parameters))
def expressions2a(IR, out, a, b, c):

    t1 = IR.var(0)
    t2 = IR.var(0)
    t3 = IR.var(0)

    for i in range(16):
        t1(i) <= a(i) ^ b(i)
        t2(i) <= a(i) ^ c(i)
        t3(i) <= b(i) ^ c(i)

        out(i) <= t1(i) | t2(i) | t3(i)

    return out

def test_2a():
    render_optm(expressions2a)

@belex_block(build_examples=lambda x : build_expressions_examples_(x, g, parameters))
def expressions3(IR, out, a, b, c):

    out() <= (a()^b()) | (a()^c()) | (b()^c())

    return out

def test_3():
    render_optm(expressions3)

@belex_block(build_examples=lambda x : build_expressions_examples_(x, g, parameters))
def expressions4(IR, out, a, b, c):
    t1 = IR.var(0)
    t2 = IR.var(0)
    t3 = IR.var(0)

    t1() <= a() ^ b()
    t2() <= a() ^ c()
    t3() <= b() ^ c()

    out() <= (a()^b()) | (a()^c()) | (b()^c())

    return out

def test_4():
    render_optm(expressions4)

@belex_block(build_examples=lambda x : build_expressions_examples_(x, g, parameters))
def expressions4a(IR, out, a, b, c):
    t1 = IR.var(0)
    t2 = IR.var(0)
    t3 = IR.var(0)

    t1() <= a() ^ b()
    t2() <= a() ^ c()
    t3() <= b() ^ c()
    for i in range(16):
        out(i) <= (a(i)^b(i)) | (a(i)^c(i)) | (b(i)^c(i))

    return out

def test_4a():
    render_optm(expressions4a)

@belex_block(build_examples=lambda x : build_expressions_examples_(x, g, parameters))
def expressions5(IR, out, a, b, c):
    t1 = IR.var(0)
    t2 = IR.var(0)
    t3 = IR.var(0)

    t1() <= a() ^ b()
    t2() <= a() ^ c()
    t3() <= b() ^ c()
    out() <= t2() |t1()|(b()^c())#| (a()^b()) | (b()^c())

    return out

def test_5():
    render_optm(expressions5)

@belex_block(build_examples=lambda x : build_expressions_examples_(x, g, parameters))
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
