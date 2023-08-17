r"""
By Dylon Edwards
"""

import numpy as np

import pytest

import hypothesis.strategies as st
from hypothesis import given

from open_belex.apl import (APL_comment, APL_masked_stmt, APL_rl_from_sb,
                            APL_rl_from_src, APL_sb_from_src, Mask)
from open_belex.expressions import Variable
from open_belex.intermediate_representation import (IrAssignOperator,
                                                    IrBinaryOperator,
                                                    access_overlap,
                                                    find_stride,
                                                    get_first_viable)

section_strategy = st.integers(min_value=0, max_value=15)


@given(data=st.data())
def test_find_stride(data):
    stride_size = data.draw(section_strategy)

    right_indices = data.draw(
        st.lists(section_strategy,
                 min_size=0,
                 max_size=16 - stride_size,
                 unique=True))
    right_indices = np.array(sorted(right_indices))
    left_indices = right_indices + stride_size

    # Determine whether to swap the indices
    if data.draw(st.booleans()):
        (left_indices, right_indices) = (right_indices, left_indices)
        stride_size = (- stride_size)

    if len(left_indices) == 0 or len(right_indices) == 0:
        assert find_stride(left_indices, right_indices) == 0
    else:
        assert find_stride(left_indices, right_indices) == stride_size

    if len(right_indices) > 1:
        right_indices = right_indices + np.arange(len(right_indices)) + 1
        right_indices = right_indices % 16
        with pytest.raises(ValueError):
            find_stride(left_indices, right_indices)


def test_ir_binary_operator_generate_apl():
    a = Variable("a")
    b = Variable("b")
    r = Variable("t_0")

    register_map = {
        a.symbol: 0,
        b.symbol: 1,
        r.symbol: 2,
    }

    # stride=0, operator=""
    # ---------------------

    a_indices = [0]
    b_indices = [0]
    r_indices = a_indices

    operator = ""
    binary_op = IrBinaryOperator(operator, a(a_indices), b(b_indices), r(r_indices))
    expected_value = [
        APL_comment(binary_op),
        APL_masked_stmt(
            msk=Mask(b_indices),
            stmt=APL_rl_from_sb(
                sbs=[register_map[b.symbol]],
                assign_op="")),
        APL_masked_stmt(
            msk=Mask(a_indices),
            stmt=APL_rl_from_sb(
                sbs=[register_map[a.symbol]],
                assign_op=operator)),
        APL_masked_stmt(
            msk=Mask(r_indices),
            stmt=APL_sb_from_src(
                sbs=[register_map[r.symbol]],
                src="RL")),
    ]
    actual_value = binary_op.generate_apl(register_map)
    assert expected_value == actual_value

    # stride=2, operator=""
    # ---------------------

    a_indices = [2, 3, 4]
    b_indices = [0, 1, 2]
    r_indices = a_indices

    operator = ""
    binary_op = IrBinaryOperator(operator, a(a_indices), b(b_indices), r(r_indices))
    expected_value = [
        APL_comment(binary_op),
        APL_masked_stmt(
            msk=Mask(b_indices),
            stmt=APL_rl_from_sb(
                sbs=[register_map[b.symbol]],
                assign_op="")),
        APL_masked_stmt(
            msk=Mask.from_hex(0xFFFF),
            stmt=APL_rl_from_src(
                src="NRL",
                assign_op="")),
        APL_masked_stmt(
            msk=Mask.from_hex(0xFFFF),
            stmt=APL_rl_from_src(
                src="NRL",
                assign_op="")),
        APL_masked_stmt(
            msk=Mask(a_indices),
            stmt=APL_rl_from_sb(
                sbs=[register_map[a.symbol]],
                assign_op=operator)),
        APL_masked_stmt(
            msk=Mask(r_indices),
            stmt=APL_sb_from_src(
                sbs=[register_map[r.symbol]],
                src="RL")),
    ]
    actual_value = binary_op.generate_apl(register_map)
    assert expected_value == actual_value

    # stride=(-3), operator=""
    # ---------------------

    a_indices = [0, 1, 2]
    b_indices = [3, 4, 5]
    r_indices = a_indices

    operator = ""
    binary_op = IrBinaryOperator(operator, a(a_indices), b(b_indices), r(r_indices))
    expected_value = [
        APL_comment(binary_op),
        APL_masked_stmt(
            msk=Mask(b_indices),
            stmt=APL_rl_from_sb(
                sbs=[register_map[b.symbol]],
                assign_op="")),
        APL_masked_stmt(
            msk=Mask.from_hex(0xFFFF),
            stmt=APL_rl_from_src(
                src="SRL",
                assign_op="")),
        APL_masked_stmt(
            msk=Mask.from_hex(0xFFFF),
            stmt=APL_rl_from_src(
                src="SRL",
                assign_op="")),
        APL_masked_stmt(
            msk=Mask.from_hex(0xFFFF),
            stmt=APL_rl_from_src(
                src="SRL",
                assign_op="")),
        APL_masked_stmt(
            msk=Mask(a_indices),
            stmt=APL_rl_from_sb(
                sbs=[register_map[a.symbol]],
                assign_op=operator)),
        APL_masked_stmt(
            msk=Mask(r_indices),
            stmt=APL_sb_from_src(
                sbs=[register_map[r.symbol]],
                src="RL")),
    ]
    actual_value = binary_op.generate_apl(register_map)
    assert expected_value == actual_value


def test_ir_assign_operator_generate_apl():
    a = Variable("a")
    b = Variable("b")

    register_map = {
        a.symbol: 0,
        b.symbol: 1,
    }

    lvalue = a("123")
    rvalue = b("456")
    assign_op = IrAssignOperator(lvalue, rvalue)
    expected_value = [
        APL_comment(assign_op),
        APL_masked_stmt(
            msk=Mask(rvalue.indices),
            stmt=APL_rl_from_sb(
                sbs=[register_map[b.symbol]],
                assign_op="")),
        APL_masked_stmt(
            msk=Mask.from_hex(0xffff),
            stmt=APL_rl_from_src(
                src='SRL',
                assign_op='')),
        APL_masked_stmt(
            msk=Mask.from_hex(0xffff),
            stmt=APL_rl_from_src(
                src='SRL',
                assign_op='')),
        APL_masked_stmt(
            msk=Mask.from_hex(0xffff),
            stmt=APL_rl_from_src(
                src='SRL',
                assign_op='')),
        APL_masked_stmt(
            msk=Mask(lvalue.indices),
            stmt=APL_sb_from_src(
                sbs=[register_map[a.symbol]],
                src="RL")),
    ]
    actual_value = assign_op.generate_apl(register_map)
    assert expected_value == actual_value


def test_access_overlap():
    t1 = Variable("t_1")
    t2 = Variable("t_2")
    assert access_overlap(t1("0"), t2("0"))
    assert not access_overlap(t1("0"), t2("1"))
    assert access_overlap(t1("135"), t2("369"))
    assert not access_overlap(t1("012"), t2("345"))


def test_get_first_viable():
    a = Variable("a")
    t = Variable("t")

    col_temps = []
    assert get_first_viable(col_temps, a("0123")) is None

    col_temps = [t("0123")]
    assert get_first_viable(col_temps, a("0123")) is None

    col_temps = [t("3456")]
    assert get_first_viable(col_temps, a("0123")) is None

    col_temps = [t("4567")]
    assert get_first_viable(col_temps, a("0123")) == t("4567")

    col_temps = [t("2345"), t("4567")]
    assert get_first_viable(col_temps, a("0123")) == t("4567")

    col_temps = [t("2345"), t("4567"), t("9ABC")]
    assert get_first_viable(col_temps, a("0123")) == t("4567")
