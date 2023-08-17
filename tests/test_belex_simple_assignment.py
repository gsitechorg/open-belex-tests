r"""
By Dylon Edwards
"""

from typing import Callable, Optional

import numpy as np

from open_belex.apl_optimizations import (delete_dead_writes,
                                          peephole_eliminate_read_after_write,
                                          peephole_replace_zero_xor)
from open_belex.bleir.types import Example, ValueParameter
from open_belex.common.constants import NUM_PLATS_PER_APUC
from open_belex.decorators import belex_block

from open_belex_tests.harness import render_bleir


def build_simple_assignment_example(value_param: Callable[[str, np.array], ValueParameter],
                          A: np.array, B: np.array,
                          expected_value: Optional[np.array] = None) -> Example:
    if expected_value is None:
        expected_value = A

    return Example(
        expected_value=value_param('out', expected_value),
        parameters=[value_param('a', A),
                    value_param('b', B)])


def build_simple_assignment_examples(value_param):
    seed = 0
    random = np.random.RandomState(seed)

    return [
        build_simple_assignment_example(value_param,
            A=np.repeat(0x0110, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0011, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_simple_assignment_example(value_param,
            A=np.repeat(0x0111, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0101, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_simple_assignment_example(value_param,
            A=np.repeat(0x08EC, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x00B2, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_simple_assignment_example(value_param,
            A=np.repeat(0xFFFE, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0001, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_simple_assignment_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_simple_assignment_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_simple_assignment_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
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

@belex_block(build_examples=build_simple_assignment_examples)
def simple_assignment(IR, out, a, b):
    c = IR.var(0)
    d = IR.var(0)

    c() <= c() ^ c()
    d() <= d() ^ d()

    out() <= c() ^ d()
    out() <= a()

    return out


# NOTE: Disabling this test because the EnsureWriteBeforeRead semantic validator
# catches that "c() <= c() ^ c()" is a read-before-write (semantic error and
# potentionally dangerous)
# ------------------------------------------------------------------------------
# def test_belex_simple_assignment_no_optim():
#     render_bleir("simple_assignment_no_opt", simple_assignment)


def test_belex_simple_assignment_with_optim():
    render_bleir("simple_assignment_opt", simple_assignment, optimizations=[
        peephole_eliminate_read_after_write,
        delete_dead_writes,
        peephole_replace_zero_xor,
    ])


if __name__ == "__main__":
    test_belex_simple_assignment_with_optim()
