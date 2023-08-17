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
            ('x', [0x0110, 25, 777, 5443]),
            ('y', [0x0011, 26, 666,12]),
            #('c', [0x2222, 33, 333,999])
        ]


#def dima_expr_1(x,y):
#    cout1 = x ^ y
#    cout0 = x & y
#    res = x ^ y ^ cout0

#    return res

#@belex_block(build_examples=lambda x : build_expressions_examples_(x, dima_expr_1, parameters))
#def dima_belex_expr_1(IR, out, x, y):

#    cout1 = IR.var(0)
#    cout0 = IR.var(0)

#    cout1() <= x() ^ y()
#    cout0() <= x() & y()

#    out() <= x() ^ y() ^ cout0()

#    return out

#def test_dima_belex_expr_1():
#    render_optm(dima_belex_expr_1)


int2vec = lambda x : [(x >> i) & 0x1 for i in range(16)]

vec2int = lambda v : sum([x*(2**i) for i,x in enumerate(v)])

def dima_expr_2(x,y):

    cout1 = int2vec(x ^ y)
    cout0 = int2vec(x & y)

    x = int2vec(x)
    y = int2vec(y)

    for i in range(1,4):
        for j in range(i,16,4):
            cout1[j] = cout1[j-1] & (x[j] ^ y[j])
    #for j in range(1,16,4):
    #    cout1[j] = cout1[j-1] #& (x[j] ^ y[j])
        #cout0[j] = x[j] & y[j] | (cout0[j-1] & (x[j]^y[j]))



    cout1 = vec2int(cout1) #x ^ y ^ cout0
    #cout0 = vec2int(cout0)

    res = cout1 #| cout0

    return res

@belex_block(build_examples=lambda x : build_expressions_examples_(x, dima_expr_2, parameters))
def dima_belex_expr_2(IR, out, x, y):

    cout1 = IR.var(0)
    cout0 = IR.var(0)

    cout1() <= x() ^ y()
    cout0() <= y() & y()

    #cout0() <= x() & y()


    #cout1("159D") <= x("159D")&y("159D")
    #for i in range(1,4):
    #    for j in range(i,16,4):
    #        cout1(j) <= cout1(j-1) #& (x(j-1)^y(j-1)) #& (x("159D")^y("159D"))
    cout1("159D") <= cout1("048C") & (x("159D")^y("159D"))

    cout1("26AE") <= cout1("159D") & (x("26AE")^y("26AE"))
    cout1("37BF") <= cout1("26AE") & (x("37BF")^y("37BF"))

    out() <= cout1() #| cout0()

    return out

def test_dima_belex_expr_2():
    render_optm(dima_belex_expr_2)


