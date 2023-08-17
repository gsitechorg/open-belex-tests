r"""
By Dylon Edwards
"""

from typing import Callable, Optional

import numpy as np

from open_belex.apl_optimizations import (delete_dead_writes,
                                          peephole_eliminate_read_after_write)
from open_belex.bleir.types import Example, ValueParameter
from open_belex.common.constants import NUM_PLATS_PER_APUC
from open_belex.decorators import belex_block

from open_belex_tests.harness import render_bleir


def build_simple_assignment3_example(value_param: Callable[[str, np.array], ValueParameter],
                          A: np.array,
                          expected_value: Optional[np.array] = None) -> Example:
    if expected_value is None:
        expected_value = A #dEEEE | d048C

    return Example(
        expected_value=value_param('out', expected_value),
        parameters=[value_param('a', A)])


def build_simple_assignment3_examples(value_param):
    seed = 0
    random = np.random.RandomState(seed)

    return [
        build_simple_assignment3_example(value_param,
            A=np.repeat(0x0110, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_simple_assignment3_example(value_param,
            A=np.repeat(0x0111, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_simple_assignment3_example(value_param,
            A=np.repeat(0x08EC, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_simple_assignment3_example(value_param,
            A=np.repeat(0xFFFE, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_simple_assignment3_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_simple_assignment3_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_simple_assignment3_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16))
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

@belex_block(build_examples=build_simple_assignment3_examples)
def simple_assignment3a(IR, out, a):
    out() <= a()

    return out

def test_belex_simple_assignment3a_with_optim():
    render_bleir("simple_assignment3a_opt", simple_assignment3a, optimizations=[
        peephole_eliminate_read_after_write,
        delete_dead_writes,
    ])


@belex_block(build_examples=build_simple_assignment3_examples)
def simple_assignment3b(IR, out, a):
    for i in range(16):
        out(i) <= a(i)

    return out

def test_belex_simple_assignment3b_with_optim():
    render_bleir("simple_assignment3b_opt", simple_assignment3b, optimizations=[
        peephole_eliminate_read_after_write,
        delete_dead_writes,
    ])


@belex_block(build_examples=build_simple_assignment3_examples)
def simple_assignment3c(IR, out, a):
    out([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) <= a([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

    return out

def test_belex_simple_assignment3c_with_optim():
    render_bleir("simple_assignment3c_opt", simple_assignment3c, optimizations=[
        peephole_eliminate_read_after_write,
        delete_dead_writes,
    ])

@belex_block(build_examples=build_simple_assignment3_examples)
def simple_assignment3d(IR, out, a):
    for i in range(4):
        out([0+i, 4+i, 8+i, 12+i]) <= a([0+i, 4+i, 8+i, 12+i])

    return out

def test_belex_simple_assignment3d_with_optim():
    render_bleir("simple_assignmentd_opt", simple_assignment3d, optimizations=[
        peephole_eliminate_read_after_write,
        delete_dead_writes,
    ])

@belex_block(build_examples=build_simple_assignment3_examples)
def simple_assignment3e(IR, out, a):
    out("048C") <= a("048C")
    out("159D") <= a("159D")
    out("26AE") <= a("26AE")
    out("37BF") <= a("37BF")

    return out

def test_belex_simple_assignment3e_with_optim():
    render_bleir("simple_assignment3e_opt", simple_assignment3e, optimizations=[
        peephole_eliminate_read_after_write,
        delete_dead_writes,
    ])


if __name__ == "__main__":
    test_belex_simple_assignment3_with_optim()
