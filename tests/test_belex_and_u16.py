r"""
By Dylon Edwards
"""

from typing import Callable, Optional

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

NUM_BITS = 16


def build_and_u16_example(value_param: Callable[[str, np.array], ValueParameter],
                          A: np.array, B: np.array,
                          expected_value: Optional[np.array] = None) -> Example:
    global NUM_BITS
    if expected_value is None:
        max_value = min(2**(NUM_BITS+1), 2**16)
        expected_value = (A & B) % max_value
    return Example(
        expected_value=value_param('out', expected_value),
        parameters=[value_param('a', A),
                    value_param('b', B)])


def build_and_u16_examples(value_param):
    return [
        build_and_u16_example(value_param,
            A=np.repeat(0x0110, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0011, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_and_u16_example(value_param,
            A=np.repeat(0x0111, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0101, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_and_u16_example(value_param,
            A=np.repeat(0x08EC, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x00B2, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_and_u16_example(value_param,
            A=np.repeat(0xFFFE, NUM_PLATS_PER_APUC).astype(np.uint16),
            B=np.repeat(0x0001, NUM_PLATS_PER_APUC).astype(np.uint16)),
    ]


@belex_block(build_examples=build_and_u16_examples)
def and_u16(IR, out, a, b):
    temp1 = IR.var()

    for i in range(16):
        temp1(i) <= a(i) & b(i)
        out(i) <= temp1(i)

    return out


def test_belex_and_u16_no_optim():
    render_bleir("and_u16_1", and_u16, optimizations=[
        peephole_eliminate_read_after_write,
        # peephole_coalesce_two_consecutive_sb_from_rl,
    ])


def test_belex_and_u16_with_optim():
    render_bleir("and_u16_2", and_u16, optimizations=[
        peephole_eliminate_read_after_write,
        delete_dead_writes,
        peephole_replace_zero_xor,
        peephole_eliminate_write_read_dependence,
        peephole_merge_rl_from_src_and_sb_from_rl,
        peephole_merge_rl_from_src_and_rl_from_sb,
        peephole_coalesce_consecutive_and_assignments,
        peephole_coalesce_two_consecutive_sb_from_src,
        peephole_coalesce_two_consecutive_sb_from_rl,
        peephole_coalesce_shift_before_op,
    ])
