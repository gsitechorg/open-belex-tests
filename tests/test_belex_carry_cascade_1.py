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

NUM_BITS = 16


def build_carry_cascade_1_example(value_param: Callable[[str, np.array], ValueParameter],
                          A: np.array, B: np.array,
                          expected_value: Optional[np.array] = None) -> Example:
    global NUM_BITS
    if expected_value is None:
        sm_048C = 0x1111

        a_FFFF = A
        b_FFFF = B

        a_048C = a_FFFF & sm_048C
        b_048C = b_FFFF & sm_048C
        c_048C = a_048C ^ b_048C

        e_FFFF = c_048C
        expected_value = e_FFFF

    return Example(
        expected_value=value_param('out', expected_value),
        parameters=[value_param('a', A),
                    value_param('b', B)])


def build_carry_cascade_1_examples(value_param):
    seed = 0
    random = np.random.RandomState(seed)

    return [
        build_carry_cascade_1_example(value_param,
            A=np.repeat(0x0110, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0011, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_carry_cascade_1_example(value_param,
            A=np.repeat(0x0111, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0101, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_carry_cascade_1_example(value_param,
            A=np.repeat(0x08EC, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x00B2, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_carry_cascade_1_example(value_param,
            A=np.repeat(0xFFFE, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0001, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_carry_cascade_1_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_carry_cascade_1_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_carry_cascade_1_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
    ]

@belex_block(build_examples=build_carry_cascade_1_examples)
def carry_cascade_1(IR, out, a, b):

    # If you uncomment the following lines, the code will work:

    # out("048C") <= a("048C") ^ b("048C")
    # return out

    c = IR.var()
    c() <= c() ^ c()

    c("048C") <= a("048C") ^ b("048C")
#    c() <= a() ^ b()
    out() <= c()

    return out


# NOTE: Disabling this test because the EnsureWriteBeforeRead semantic validator
# catches that "c() <= c() ^ c()" is a read-before-write (semantic error and
# potentionally dangerous)
# ------------------------------------------------------------------------------
# def test_belex_carry_cascade_1_no_optim():
#     render_bleir("carry_cascade_1_no_opt", carry_cascade_1)


def test_belex_carry_cascade_1_with_optim():
    render_bleir("carry_cascade_1_opt", carry_cascade_1, optimizations=[
        peephole_eliminate_read_after_write,
        delete_dead_writes,
        peephole_replace_zero_xor,
    ])


if __name__ == "__main__":
    test_belex_carry_cascade_1_with_optim()
