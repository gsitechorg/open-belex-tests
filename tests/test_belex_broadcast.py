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
            ('x', [422, 485, 679, 615, 434, 695, 984, 900, 671, 740, 10, 826, 139, 981, 632]),
            ('y', [336, 600, 285, 392, 134, 659, 586, 601, 29, 368, 295, 925, 381, 901, 705]),
            #('c', [0x2222, 33, 333,999])
        ]

int2vec = lambda x : [(x >> i) & 0x1 for i in range(16)]

vec2int = lambda v : sum([x*(2**i) for i,x in enumerate(v)])

def broadcast_2(x,y):

    res = y^y
    j = 2
    i = 0x001f

    val = ((x & (0x1 << j)) >> j)
    res = (res & ~i) | ((i)*val)

    return res

@belex_block(build_examples=lambda x : build_expressions_examples_(x, broadcast_2, parameters))
def dima_belex_broadcast_2(IR, out, x, y):

    out() <= y()^y()

    out([0,1,2,3,4]) <= x(2)
    return out

def test_dima_belex_expr_2():
    render_optm(dima_belex_broadcast_2)


