r"""
By Dylon Edwards
"""

from typing import Callable, List, Optional

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
from open_belex.utils.config_utils import belex_config
from open_belex_tests.harness import render_bleir

NUM_BITS = 16


def build_add_u16_example(value_param: Callable[[str, np.array], ValueParameter],
                          A: np.array,
                          B: np.array,
                          expected_value: Optional[np.array] = None) -> Example:

    r"""Return an `Example`, which contains (1) a singleton
    `ValueParameter` for the `expected_value` of a FUT (Function
    Under Test); and (2) a `Sequence` of `ValueParameters`, one
    for each of the input values of the FUT. For example, if the
    FUT is `add` two VRs, `a` and `b`, then the `Example` contains
    a singleton `ValueParameter` for the resulting VR containing
    the sum, plus two `ValueParameters` in a `Sequence`, one
    `ValueParameter` for the input VR `a` and another
    `ValueParameter` for the input VR `b`.

    Each VR has the shape of a numpy array of 16 X 2048 bits. To
    perform the test, the test harness will compare the expected
    value to the actual output of the FUT.

    If the `expected_value` parameter of this function,
    `build_add_u16_example`, is None, then this function will
    compute the sum of the inputs `a` and `b` via ordinary Python.
    That sum is regarded as the ground truth for the FUT. Thus,
    this function is not truly general, but is specialized for the
    sum. This fact is reflected in the name,
    build_ADD_u16_examples.

    """

    global NUM_BITS

    mask = 2 ** NUM_BITS - 1

    if expected_value is None:
        expected_value = (A + B) & mask

    return Example(
        expected_value=value_param('out', expected_value),
        parameters=[value_param('a', A & mask),
                    value_param('b', B & mask)])


def build_add_u16_examples(value_param) -> List[Example]:

    r"""Return a list of examples constructed by
    `build_add_u16_example`."""

    seed = 0
    random = np.random.RandomState(seed)

    return [
        build_add_u16_example(value_param,
            A=np.repeat(0x0110, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0011, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_add_u16_example(value_param,
            A=np.repeat(0x0111, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0101, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_add_u16_example(value_param,
            A=np.repeat(0x08EC, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x00B2, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_add_u16_example(value_param,
            A=np.repeat(0xFFFE, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0001, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_add_u16_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF,
                             size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF,
                             size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_add_u16_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF,
                             size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF,
                             size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_add_u16_example(value_param,
            A=random.randint(low=0x0000, high=0xFFFF,
                             size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
            B=random.randint(low=0x0000, high=0xFFFF,
                             size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
    ]


@belex_block(build_examples=build_add_u16_examples)
def add_u16_dnf(IR, out, a, b):
    r"""Ripple-carry adder in Disjunctive Normal Form (DNF)."""
    global NUM_BITS

    carry = IR.var()
    temp = IR.var()

    out(0) <= a(0) ^ b(0)
    carry(0) <= a(0) & b(0)
    for i in range(1, NUM_BITS):
        temp(i) <= a(i) & b(i)
        carry(i) <= temp(i) | (a(i) & carry(i - 1)) | (b(i) & carry(i - 1))
        # FIXME: The vectorizer currently cannot handle the following
        # expression, which is why we're using the temp(i) above. The code
        # passes but it is repeated on each iteration of the loop (not
        # vectorized).
        # -------------------------------------------------------------------------
        # carry(i) <= (a(i) & b(i)) | (a(i) & carry(i - 1)) | (b(i) & carry(i - 1))
        out(i) <= a(i) ^ b(i) ^ carry(i-1)

    return out


# @pretty_print("add_u16_cnf", colorize=False)
@belex_block(build_examples=build_add_u16_examples)
def add_u16_cnf(IR, out, a, b):
    r"""Ripple-carry adder in Conjunctive Normal Form (CNF)."""
    global NUM_BITS

    carry = IR.var()
    temp = IR.var()

    out[0] <= a(0) ^ b(0)
    carry[0] <= a(0) & b(0)
    for i in range(1, NUM_BITS):
        temp[i] <= a(i) | b(i)
        carry[i] <= temp(i) & (a(i) | carry(i - 1)) & (b(i) | carry(i - 1))
        out[i] <= a(i) ^ b(i) ^ carry(i-1)

    return out


def test_belex_add_u16_no_optim():
    render_bleir("add_u16_dnf_1", add_u16_dnf)
    render_bleir("add_u16_cnf_1", add_u16_cnf)


OPTIMIZATIONS = [
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
]


def test_belex_add_u16_with_optim():
    render_bleir("add_u16_dnf_2", add_u16_dnf, optimizations=OPTIMIZATIONS)
    render_bleir("add_u16_cnf_2", add_u16_cnf, optimizations=OPTIMIZATIONS)


@belex_config(max_rn_regs=5)
def test_belex_add_u16_with_optim_w_5_rn_regs():
    render_bleir("add_u16_dnf_3", add_u16_dnf, optimizations=OPTIMIZATIONS)
    render_bleir("add_u16_cnf_3", add_u16_cnf, optimizations=OPTIMIZATIONS)


@belex_config(max_rn_regs=3)
def test_belex_add_u16_with_optim_w_3_rn_regs():
    optimizations = [
        delete_dead_writes,
        peephole_eliminate_read_after_write,
    ]
    render_bleir("add_u16_dnf_4", add_u16_dnf, optimizations=optimizations)
    render_bleir("add_u16_cnf_4", add_u16_cnf, optimizations=optimizations)


if __name__ == "__main__":
    test_belex_add_u16_with_optim()
