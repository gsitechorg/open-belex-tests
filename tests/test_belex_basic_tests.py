r"""
By Dylon Edwards
"""

from typing import Callable, Optional

import numpy as np

import pytest

from open_belex.apl_optimizations import (
    delete_dead_writes, peephole_coalesce_consecutive_and_assignments,
    peephole_coalesce_shift_before_op,
    peephole_coalesce_two_consecutive_sb_from_rl,
    peephole_coalesce_two_consecutive_sb_from_src,
    peephole_eliminate_read_after_write,
    peephole_eliminate_write_read_dependence,
    peephole_merge_rl_from_src_and_rl_from_sb,
    peephole_merge_rl_from_src_and_sb_from_rl, peephole_replace_zero_xor)
from open_belex.bleir.types import Example, SemanticError, ValueParameter
from open_belex.common.constants import NUM_PLATS_PER_APUC
from open_belex.decorators import belex_block
from open_belex_tests.harness import render_bleir

NUM_BITS = 16


def build_example(value_param: Callable[[str, np.array], ValueParameter],
                  f,
                  A: np.array,
                  B: np.array,
                  expected_value: Optional[np.array] = None) -> Example:
    if expected_value is None:
        expected_value = f(A, B) % min(2 ** (NUM_BITS + 1), 2 ** 16)
    return Example(
        expected_value=value_param('out', expected_value),
        parameters=[value_param('a', A),
                    value_param('b', B)])


def build_examples(value_param, f):
    seed = 0
    random = np.random.RandomState(seed)

    return [
        build_example(value_param, f,
                      A=np.repeat(0x0110, NUM_PLATS_PER_APUC).astype(np.uint16),
                      B=np.repeat(0x0011, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_example(value_param, f,
                      A=np.repeat(0x0111, NUM_PLATS_PER_APUC).astype(np.uint16),
                      B=np.repeat(0x0101, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_example(value_param, f,
                      A=np.repeat(0x08EC, NUM_PLATS_PER_APUC).astype(np.uint16),
                      B=np.repeat(0x00B2, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_example(value_param, f,
                      A=np.repeat(0xFFFE, NUM_PLATS_PER_APUC).astype(np.uint16),
                      B=np.repeat(0x0001, NUM_PLATS_PER_APUC).astype(np.uint16)),
        build_example(value_param, f,
                      A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
                      B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_example(value_param, f,
                      A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
                      B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
        build_example(value_param, f,
                      A=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16),
                      B=random.randint(low=0x0000, high=0xFFFF, size=(NUM_PLATS_PER_APUC,), dtype=np.uint16)),
    ]


def render_code(belex_code, optims=[
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
]):
    snippet_name = f"func_{belex_code.__name__}"
    render_bleir(snippet_name, belex_code, optimizations=optims)


#
# TESTS
#

# xor


## this should pass
@belex_block(build_examples=lambda x: build_examples(x, lambda a, b: a ^ b))
def xor_u16a(IR, out, a, b):
    temp1 = IR.var()

    for i in range(16):
        out(i) <= a(i) ^ b(i)
        # out(i) <= temp1(i)

    return out


def test_belex_xor_u16a_with_optim():
    render_code(xor_u16a)


@belex_block(build_examples=lambda x: build_examples(x, lambda a, b: a & b))
def and_u16a(IR, out, a, b):
    temp1 = IR.var()

    for i in range(16):
        out(i) <= a(i) & b(i)

    return out


def test_belex_and_u16a_with_optim():
    render_code(and_u16a)


## already vectorized version
@belex_block(build_examples=lambda x: build_examples(x, lambda a, b: a & b))
def and_u16b(IR, out, a, b):
    # temp1 = IR.var()
    out([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) <= a(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) & b(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    # for i in range(16):
    #    out(i) <= a(i) & b(i)

    return out


def test_belex_and_u16b_with_optim():
    render_code(and_u16b)

    ## already vectorized version


@belex_block(build_examples=lambda x: build_examples(x, lambda a, b: a & b))
def and_u16c(IR, out, a, b):
    all_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # temp1 = IR.var()
    out(all_) <= a(all_) & b(all_)
    # for i in range(16):
    #    out(i) <= a(i) & b(i)

    return out


def test_belex_and_u16c_with_optim():
    render_code(and_u16c)


@belex_block(build_examples=lambda x: build_examples(x, lambda a, b: (a & b) & 0x000f))
def and_u16d(IR, out, a, b):
    all_ = list(range(16))

    out(all_) <= out(all_) ^ out(all_)

    for i in range(4):
        out(i) <= a(i) & b(i)

    return out


def test_belex_and_u16d_with_optim():
    render_code(and_u16d)


@belex_block(build_examples=lambda x: build_examples(x, lambda a, b: (a & b) & 0x000f))
def and_u16e(IR, out, a, b):
    all_ = list(range(4, 16))

    out(all_) <= out(all_) ^ out(all_)

    for i in range(4):
        out(i) <= a(i) & b(i)

    return out


def test_belex_and_u16e_with_optim():
    render_code(and_u16e)


def test_belex_xor_u16c_with_optim():

    @belex_block(build_examples=lambda x: build_examples(x, lambda a, b: a ^ b),
                 should_fail=True)
    def xor_u16c(IR, out, a, b):
        temp1 = IR.var()

        for i in range(16):
            out(i) <= a(i) ^ b(i)
            out(i) <= temp1(i)

        return out

    with pytest.raises(SemanticError):
        render_code(xor_u16c)


# this should pass
@belex_block(build_examples=lambda x: build_examples(x, lambda a, b: a ^ b))
def xor_u16b(IR, out, a, b):
    temp1 = IR.var()

    for i in range(16):
        temp1(i) <= a(i) ^ b(i)
        out(i) <= temp1(i)

    return out


def test_belex_xor_u16b_with_optim():
    render_code(xor_u16b)


@belex_block(build_examples=lambda x: build_examples(x, lambda a, b: (a ^ b) | a))
def xor_u16(IR, out, a, b):
    temp1 = IR.var()

    for i in range(16):
        temp1(i) <= a(i) ^ b(i)
        out(i) <= temp1(i) | a(i)

    return out


def test_belex_xor_u16_with_optim():
    render_code(xor_u16)


@belex_block(build_examples=lambda x: build_examples(x, lambda a, b: a + b))
def add_u16(IR, out, a, b):
    carry = IR.var()
    temp = IR.var()

    out(0) <= a(0) ^ b(0)
    carry(0) <= a(0) & b(0)
    for i in range(1, NUM_BITS):
        temp(i) <= a(i) & b(i)
        carry(i) <= temp(i) | a(i) & carry(i - 1) | b(i) & carry(i - 1)
        out(i) <= a(i) ^ b(i) ^ carry(i - 1)

    return out


def test_belex_add_u16_with_optim():
    render_code(add_u16, optims=[
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
