r"""
By Dylon Edwards and Brian Beckman
"""


import numpy as np

import hypothesis
from hypothesis import given

from open_belex.diri.half_bank import DIRI
from open_belex.literal import (GGL, GL, NRL, RL, RN_REG_FLAGS, RN_REG_T0,
                                RN_REG_T1, SM_0X000F, SM_0X0001, SM_0X1111,
                                SM_0X3333, SM_0XFFFF, VR, belex_apl)
from open_belex.utils.example_utils import convert_to_bool, convert_to_u16

from open_belex_libs.arithmetic import (
    add_u16, add_u16_lifted_rn_regs, add_u16_lifted_rn_regs_all_lifted_sm_regs,
    add_u16_lifted_rn_regs_one_lifted_sm_reg, add_u16_literal_sections,
    sub_u16)
from open_belex_libs.bitwise import xor_16
from open_belex_libs.common import cpy_imm_16

from open_belex_tests.utils import (belex_property_test,
                                    parameterized_belex_test, vr_strategy)

#    ___                       ___            _ _    _   _
#   / __|__ _ _ _ _ _ _  _ ___| _ \_ _ ___ __| (_)__| |_(_)___ _ _
#  | (__/ _` | '_| '_| || |___|  _/ '_/ -_) _` | / _|  _| / _ \ ' \
#   \___\__,_|_| |_|  \_, |   |_| |_| \___\__,_|_\__|\__|_\___/_||_|
#    /_\  __| |__| |__|__/_
#   / _ \/ _` / _` / -_) '_|
#  /_/ \_\__,_\__,_\___|_|


@parameterized_belex_test
def test_add_u16_w_carry_pred(diri: DIRI) -> int:
    r"""Test Moshe's adder with symbolic section masks. Assert
    that generated code is identical to the hand-written version."""

    res_vp = 0
    x_vp = 1
    y_vp = 2

    a_vp = 0x2000
    b_vp = 0x3000
    expected_value = 0x5000

    cpy_imm_16(x_vp, a_vp)
    cpy_imm_16(y_vp, b_vp)

    fragment_caller_call = add_u16(res_vp, x_vp, y_vp)
    fragment = fragment_caller_call.fragment

    x_ = convert_to_u16(diri.hb[x_vp])
    y_ = convert_to_u16(diri.hb[y_vp])
    r_ = convert_to_u16(diri.hb[res_vp])

    assert all(r_ == x_ + y_)

    assert "\n".join(map(str, fragment.operations)) == "\n".join([
        "{SM_0XFFFF: RL = SB[x];}",
        "{SM_0XFFFF: RL ^= SB[y]; "
         "SM_0X3333: GGL = RL;}",
        "{SM_0XFFFF: SB[RN_REG_T0] = RL;}",
        "{SM_0X1111: SB[RN_REG_T1] = RL; "
         "SM_0X1111<<1: SB[RN_REG_T1] = GGL; "
         "SM_0X1111<<2: RL = SB[RN_REG_T0] & GGL; "
         "SM_0X3333: RL = SB[x,y];}",
        "{SM_0X1111<<2: SB[RN_REG_T1] = RL; "
         "SM_0X1111<<3: RL = SB[RN_REG_T0] & NRL; "
         "SM_0X1111<<1: RL |= SB[RN_REG_T0] & NRL; "
         "SM_0X1111<<2: RL = SB[x,y];}",
        "{SM_0X1111<<3: SB[RN_REG_T1] = RL; "
         "SM_0X1111<<3: RL = SB[x,y]; "
         "SM_0X1111<<2: RL |= SB[RN_REG_T0] & NRL; "
         "SM_0X0001: GGL = RL;}",
        "{SM_0X1111<<3: RL |= SB[RN_REG_T0] & NRL; "
         "SM_0X0001<<3: GL = RL; "
         "SM_0X0001: RL = SB[RN_REG_T1];}",
        "{SM_0X000F<<4: RL |= SB[RN_REG_T1] & GL; "
         "SM_0X0001<<7: GL = RL; "
         "SM_0X0001: SB[res] = RL;}",
        "{SM_0X000F<<8: RL |= SB[RN_REG_T1] & GL; "
         "SM_0X0001<<11: GL = RL; "
         "SM_0X0001: RL = GGL;}",
        "{SM_0X000F<<12: RL |= SB[RN_REG_T1] & GL; "
         "SM_0X0001<<15: GL = RL;}",
        "{SM_0X0001: SB[RN_REG_FLAGS] = GL; "
         "~SM_0X0001: RL = SB[RN_REG_T0] ^ NRL;}",
        "{~SM_0X0001: SB[res] = RL;}",
    ])

    return res_vp


@belex_property_test(add_u16)
def test_random_add_u16_w_carry_pred(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Test Moshe's adder with symbolic sections masks and
    random data."""
    return x + y


@belex_property_test(sub_u16)
def test_random_sub_u16(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x - y


@parameterized_belex_test
def test_add_u16_literal_section_w_carry_pred(diri: DIRI) -> int:
    r"""Test Moshe's adder with literal section lists. Assert
    that generated code is identical to the hand-written version."""
    res_vp = 0
    x_vp = 1
    y_vp = 2

    a_vp = 0x2000
    b_vp = 0x3000
    expected_value = 0x5000

    cpy_imm_16(x_vp, a_vp)
    cpy_imm_16(y_vp, b_vp)

    fragment_caller_call = add_u16_literal_sections(res_vp, x_vp, y_vp)
    fragment = fragment_caller_call.fragment

    assert "\n".join(map(str, fragment.operations)) == "\n".join([
        "{_INTERNAL_SM_0XFFFF: RL = SB[x];}",
        "{_INTERNAL_SM_0XFFFF: RL ^= SB[y]; "
         "_INTERNAL_SM_0X3333: GGL = RL;}",
        "{_INTERNAL_SM_0XFFFF: SB[RN_REG_T0] = RL;}",
        "{_INTERNAL_SM_0X1111: SB[RN_REG_T1] = RL; "
         "_INTERNAL_SM_0X1111<<1: SB[RN_REG_T1] = GGL; "
         "_INTERNAL_SM_0X1111<<2: RL = SB[RN_REG_T0] & GGL; "
         "_INTERNAL_SM_0X3333: RL = SB[x,y];}",
        "{_INTERNAL_SM_0X1111<<2: SB[RN_REG_T1] = RL; "
         "_INTERNAL_SM_0X1111<<3: RL = SB[RN_REG_T0] & NRL; "
         "_INTERNAL_SM_0X1111<<1: RL |= SB[RN_REG_T0] & NRL; "
         "_INTERNAL_SM_0X1111<<2: RL = SB[x,y];}",
        "{_INTERNAL_SM_0X1111<<3: SB[RN_REG_T1] = RL; "
         "_INTERNAL_SM_0X1111<<3: RL = SB[x,y]; "
         "_INTERNAL_SM_0X1111<<2: RL |= SB[RN_REG_T0] & NRL; "
         "_INTERNAL_SM_0X0001: GGL = RL;}",
        "{_INTERNAL_SM_0X1111<<3: RL |= SB[RN_REG_T0] & NRL; "
         "_INTERNAL_SM_0X0001<<3: GL = RL; "
         "_INTERNAL_SM_0X0001: RL = SB[RN_REG_T1];}",
        "{_INTERNAL_SM_0X000F<<4: RL |= SB[RN_REG_T1] & GL; "
         "_INTERNAL_SM_0X0001<<7: GL = RL; "
         "_INTERNAL_SM_0X0001: SB[res] = RL;}",
        "{_INTERNAL_SM_0X000F<<8: RL |= SB[RN_REG_T1] & GL; "
         "_INTERNAL_SM_0X0001<<11: GL = RL; "
         "_INTERNAL_SM_0X0001: RL = GGL;}",
        "{_INTERNAL_SM_0XFFFF<<12: RL |= SB[RN_REG_T1] & GL; "
         "_INTERNAL_SM_0XFFFF<<15: GL = RL;}",
        "{_INTERNAL_SM_0X0001: SB[RN_REG_FLAGS] = GL; "
         "_INTERNAL_SM_0XFFFF<<1: RL = SB[RN_REG_T0] ^ NRL;}",
        "{_INTERNAL_SM_0XFFFF<<1: SB[res] = RL;}",
    ])

    return res_vp


@belex_property_test(add_u16_literal_sections)
def test_random_add_u16_literal_sections_w_carry_pred(
        x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Test Moshe's adder with symbolic sections masks and
    random data."""
    return x + y


# @pytest.mark.skip("property test stubbed for now.")
# @belex_property_test(add_u16_lifted_rn_regs)
# def test_random_add_u16_lifted_rn_regs_property(
#         """Test fails because test infra sometimes assigns RN=15 to flags!"""
#         x: np.ndarray,
#         y: np.ndarray,
#         x_xor_y: np.ndarray,
#         cout1: np.ndarray,
#         flags: np.ndarray) -> np.ndarray:
#     r"""test Moshe's adder with lifted RN_REGs"""
#     return (x + y) & 0xFFFF


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_random_add_u16_lifted_rn_regs(d: DIRI):
    r"""TODO: This test compiles, runs, and fails if and only if
    the flags register is set to 15."""
    x = 1
    y = 2
    res = 0
    x_xor_y = 8
    cout1 =   9
    flags =  14  #ANYTHING EXCEPT 15! RN_REG_FLAGS.register  # 15

    add_u16_lifted_rn_regs(res, x, y, x_xor_y, cout1, flags)

    x_ = convert_to_u16(d.hb[x])
    y_ = convert_to_u16(d.hb[y])
    r_ = convert_to_u16(d.hb[res])
    xXy_ = convert_to_u16(d.hb[x_xor_y])

    assert all(r_ == x_ + y_)
    assert all(xXy_ == x_ ^ y_)

    return res


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_random_add_u16_lifted_rn_regs_one_lifted_sm_reg(d: DIRI):
    r"""TODO: This test compiles, runs, and fails if and only if
    the flags register is set to 15."""
    x = 1
    y = 2
    res = 0
    x_xor_y = 8
    cout1 =   9
    flags =  23  #ANYTHING EXCEPT 15! RN_REG_FLAGS.register  # 15

    sm_just_one = 0x0001

    add_u16_lifted_rn_regs_one_lifted_sm_reg(res, x, y, x_xor_y, cout1, flags, sm_just_one)

    x_ = convert_to_u16(d.hb[x])
    y_ = convert_to_u16(d.hb[y])
    r_ = convert_to_u16(d.hb[res])
    xXy_ = convert_to_u16(d.hb[x_xor_y])

    assert all(r_ == x_ + y_)
    assert all(xXy_ == x_ ^ y_)

    return res


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_random_add_u16_lifted_rn_regs_all_lifted_sm_regs(d: DIRI):
    r"""TODO: This test compiles, runs, and fails if and only if
    the flags register is set to 15."""
    x = 1
    y = 2
    res = 0
    x_xor_y = 8
    cout1 =   9
    flags =  23  #ANYTHING EXCEPT 15! RN_REG_FLAGS.register  # 15

    sm_just_one = 0x0001

    add_u16_lifted_rn_regs_all_lifted_sm_regs(
        res, x, y, x_xor_y, cout1, flags,
        sm_just_one, sm_all=0xffff,
        sm_threes=0x3333, sm_ones=0x1111,
        sm_one_f=0x000f)

    x_ = convert_to_u16(d.hb[x])
    y_ = convert_to_u16(d.hb[y])
    r_ = convert_to_u16(d.hb[res])
    xXy_ = convert_to_u16(d.hb[x_xor_y])

    assert all(r_ == x_ + y_)
    assert all(xXy_ == x_ ^ y_)

    return res


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_random_add_u16_non_lifted_rn_regs(d: DIRI):
    r"""Compare with lifted version"""
    x = 1
    y = 2
    res = 3

    fcc = add_u16(res, x, y)

    x_ = convert_to_u16(d.hb[x])
    y_ = convert_to_u16(d.hb[y])
    r_ = convert_to_u16(d.hb[res])

    assert all(r_ == x_ + y_)

    return res


@belex_apl
def add_u16_no_lanes(Belex, res: VR, x: VR, y: VR) -> None:
    r"""Moshe's 12-clock carry-prediction adder, verbatim in BELEX."""
    os = SM_0X0001
    fs = SM_0XFFFF
    threes = SM_0X3333
    ones = SM_0X1111
    one_f = SM_0X000F

    x_xor_y = RN_REG_T0
    cout1 = RN_REG_T1
    flags = RN_REG_FLAGS

    # Carry in/out flag
    C_FLAG = 0

    RL[fs] <= x()
    RL[fs] ^= y()
    GGL[threes] <= RL()
    x_xor_y[fs] <= RL()
    cout1[ones] <= RL()
    cout1[ones<<1] <= GGL()
    RL[ones<<2] <= x_xor_y() & GGL()
    RL[threes] <= x() & y()
    cout1[ones<<2] <= RL()
    RL[ones<<3] <= x_xor_y() & NRL()
    RL[ones<<1] |= x_xor_y() & NRL()
    RL[ones<<2] <= x() & y()
    cout1[ones<<3] <= RL()
    RL[ones<<3] <= x() & y()
    RL[ones<<2] |= x_xor_y() & NRL()
    GGL[os] <= RL()
    RL[ones<<3] |= x_xor_y() & NRL()
    GL[os<<3] <= RL()
    RL[os] <= cout1()
    RL[one_f<<4] |= cout1() & GL()
    GL[os<<7] <= RL()
    res[os] <= RL()
    RL[one_f<<8] |= cout1() & GL()
    GL[os<<11] <= RL()
    RL[os] <= GGL()
    RL[one_f<<12] |= cout1() & GL()
    GL[os<<15] <= RL()
    flags[os<<C_FLAG] <= GL()
    RL[~os] <= x_xor_y() ^ NRL()
    res[~os] <= RL()


@hypothesis.settings(max_examples=5, deadline=None)
@given(a=vr_strategy(), b=vr_strategy())
@parameterized_belex_test
def test_add_u16_no_lanes(diri: DIRI, a: np.ndarray, b: np.ndarray) -> int:
    out_vp = 0
    tmp_vp = 1
    a_vp = 2
    b_vp = 3

    diri.hb[a_vp, ::, ::] = convert_to_bool(a)
    diri.hb[b_vp, ::, ::] = convert_to_bool(b)

    add_u16_no_lanes(tmp_vp, a_vp, b_vp)
    add_u16(out_vp, a_vp, b_vp)
    xor_16(out_vp, out_vp, tmp_vp)

    assert not diri.hb[out_vp].any()

    return out_vp


#  ___      _    _               _
# / __|_  _| |__| |_ _ _ __ _ __| |_ ___ _ _
# \__ \ || | '_ \  _| '_/ _` / _|  _/ _ \ '_|
# |___/\_,_|_.__/\__|_| \__,_\__|\__\___/_|


@belex_property_test(add_u16)
def test_random_add_u16_w_carry_pred(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Test Moshe's adder with symbolic sections masks and
    random data."""
    return x + y


# Multiplicator

# NOTE: Unsafe test: run manually.
# -> There is a conflict in WRITE SBs in the 2nd instruction of
# -> mul_u16_u16xu16_7t (z_lsb and y) according to
# -> MultiStatementSBGroupingEnforcer. I either do not fully understand the
# -> laning rules of WRITEs or z_lsb has to be chosen very carefully. If the
# -> semantic checker is disabled, this test passes.

# @hypothesis.settings(max_examples=5, deadline=None)
# @given(x=vr_strategy(), y=vr_strategy())
# @parameterized_belex_test
# def test_mul_u16(diri: DIRI, x: np.ndarray, y: np.ndarray) -> int:
#     z_vp = 0
#     x_vp = 1
#     y_vp = 2

#     diri.hb[x_vp, ::, ::] = convert_to_bool(x)
#     diri.hb[y_vp, ::, ::] = convert_to_bool(y)

#     expected_z = convert_to_bool((x * y) & 0xFFFF)

#     apl_init()
#     mul_u16(z_vp, x_vp, y_vp)

#     actual_z = diri.hb[z_vp]
#     assert np.array_equal(expected_z, actual_z)

#     return z_vp
