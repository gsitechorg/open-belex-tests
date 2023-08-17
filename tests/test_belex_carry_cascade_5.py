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


def build_carry_cascade_5_example(value_param: Callable[[str, np.array], ValueParameter],
                          A: np.array, B: np.array,
                          expected_value: Optional[np.array] = None) -> Example:
    if expected_value is None:
        sm_048C = 0x1111
        sm_159D = 0x2222
        sm_FFFF = 0xFFFF

        aFFFF = A
        bFFFF = B

        dFFFF = aFFFF

        a048C = aFFFF & sm_048C
        b159D = bFFFF & sm_159D
        d048C = a048C ^ ((b159D >> 1) & sm_FFFF)

        dFFFF = (dFFFF & ~sm_048C) | d048C
        expected_value = dFFFF

    return Example(
        expected_value=value_param('out', expected_value),
        parameters=[value_param('a', A),
                    value_param('b', B)])


def build_carry_cascade_5_examples(value_param):
    seed = 0
    random = np.random.RandomState(seed)

    return [
        build_carry_cascade_5_example(value_param,
            A=np.repeat(0x0110, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0011, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_carry_cascade_5_example(value_param,
            A=np.repeat(0x0111, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0101, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_carry_cascade_5_example(value_param,
            A=np.repeat(0x08EC, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x00B2, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_carry_cascade_5_example(value_param,
            A=np.repeat(0xFFFE, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0001, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_carry_cascade_5_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_carry_cascade_5_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_carry_cascade_5_example(value_param,
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

@belex_block(build_examples=build_carry_cascade_5_examples)
def carry_cascade_5(IR, out, a, b):
    d = IR.var(0)  # _INTERNAL0

    d() <= a() # fail if uncomment ^ b()
    d("048C") <= a("048C") ^ b("159D")

    out() <= d()

    return out


def test_belex_carry_cascade_5_no_optim():
    render_bleir("carry_cascade_5_no_opt", carry_cascade_5)


def test_belex_carry_cascade_5_with_optim():
    render_bleir("carry_cascade_5_opt", carry_cascade_5, optimizations=[
        peephole_eliminate_read_after_write,
        delete_dead_writes,
    ])


if __name__ == "__main__":
    test_belex_carry_cascade_5_with_optim()
