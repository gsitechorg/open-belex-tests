r"""
By Dylon Edwards and Brian Beckman
"""


import numpy as np

import hypothesis
from hypothesis import given

from open_belex.diri.half_bank import DIRI
from open_belex.literal import (ERL, GGL, GL, INV_ERL, INV_GGL, INV_GL,
                                INV_NRL, INV_RL, INV_RSP16, INV_SRL, INV_WRL,
                                NOOP, NRL, RL, RN_REG_T0, RN_REG_T3, RN_REG_T4,
                                RN_REG_T5, RN_REG_T6, RSP16, RSP_END,
                                RSP_START_RET, SM_0XFFFF, SRL, VR, WRL, Mask,
                                apl_commands, belex_apl, u16)
from open_belex.utils.example_utils import convert_to_u16, u16_to_vr

from open_belex_libs.bitwise import and_16, or_16, xor_16
from open_belex_libs.common import (cpy_imm_16, cpy_vr, rl_from_sb, rsp_out,
                                    rsp_out_in)
from open_belex_libs.memory import load_16, store_16, swap_vr_vmr_16
from open_belex_libs.tartan import (_write_test_markers, tartan_and_equals,
                                    tartan_assign, tartan_imm_donor,
                                    tartan_or_equals, tartan_xor_equals,
                                    write_to_marked)

from open_belex_tests.utils import (Mask_strategy, belex_property_test,
                                    parameterized_belex_test, u16_strategy)

#                  _               _  __
#   __ _ __ _  _  (_)_ __  _ __   / |/ /
#  / _| '_ \ || | | | '  \| '  \  | / _ \
#  \__| .__/\_, |_|_|_|_|_|_|_|_|_|_\___/
#     |_|   |__/___|           |___|


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_cpy_imm_16(diri: DIRI) -> int:
    tgt_vp = 0
    val_vp = 0xBEEF
    fragment_caller_call = cpy_imm_16(tgt_vp, val_vp)
    fragment = fragment_caller_call.fragment

    assert "\n".join(map(str, fragment.operations)) == "\n".join([
        "{val: SB[tgt] = INV_RSP16; "
         "~val: SB[tgt] = RSP16;}",
    ])

    tgt_vr = convert_to_u16(diri.hb[tgt_vp])
    assert all(tgt_vr == val_vp)


@hypothesis.settings(max_examples=3, deadline=None)
@given(val=u16_strategy(min_value=0x0001))
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_random_cpy_imm_16(diri: DIRI, val: int) -> int:
    tgt_sb = 0
    tgt_vr = u16_to_vr(val)
    cpy_imm_16(tgt_sb, val)
    assert np.array_equal(tgt_vr, convert_to_u16(diri.SB[tgt_sb]))
    return tgt_sb


@belex_apl
def cpy_imm_16_2x(Belex, res1: VR, res2: VR, val: u16) -> None:
    # Demonstrates function inlining
    cpy_imm_16(res1, val)
    cpy_imm_16(res2, val)


@hypothesis.settings(max_examples=3, deadline=None)
@given(val=u16_strategy(min_value=0x0001))
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_cpy_imm_16_2x(diri: DIRI, val: int) -> np.ndarray:
    res1_sb = 0
    res1_vr = u16_to_vr(val)

    res2_sb = 1
    res2_vr = u16_to_vr(val)

    cpy_imm_16_2x(res1_sb, res2_sb, val)
    assert np.array_equal(res1_vr, convert_to_u16(diri.SB[res1_sb]))
    assert np.array_equal(res2_vr, convert_to_u16(diri.SB[res2_sb]))
    return res1_sb


@belex_apl
def cpy_imm_16_3x(Belex, res1: VR, res2: VR, res3: VR, val: u16) -> None:
    # Demonstrates function inlining
    cpy_imm_16_2x(res1, res2, val)
    cpy_imm_16(res3, val)


@hypothesis.settings(max_examples=3, deadline=None)
@given(val=u16_strategy(min_value=0x0001))
@parameterized_belex_test(
    repeatably_randomize_half_bank=True)
def test_cpy_imm_16_3x(diri: DIRI, val: int) -> np.ndarray:
    res1_sb = 0
    res1_vr = u16_to_vr(val)

    res2_sb = 1
    res2_vr = u16_to_vr(val)

    res3_sb = 2
    res3_vr = u16_to_vr(val)

    cpy_imm_16_3x(res1_sb, res2_sb, res3_sb, val)
    assert np.array_equal(res1_vr, convert_to_u16(diri.SB[res1_sb]))
    assert np.array_equal(res2_vr, convert_to_u16(diri.SB[res2_sb]))
    assert np.array_equal(res3_vr, convert_to_u16(diri.SB[res3_sb]))
    return res1_sb


@belex_apl
def cpy_imm_16_4x(Belex, res1: VR, res2: VR, res3: VR, res4: VR, val: u16) -> None:
    # Demonstrates function inlining
    cpy_imm_16_3x(res1, res2, res3, val)
    cpy_imm_16(res4, val)


@hypothesis.settings(max_examples=3, deadline=None)
@given(val=u16_strategy(min_value=0x0001))
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_cpy_imm_16_4x(diri: DIRI, val: int) -> np.ndarray:
    res1_sb = 0
    res1_vr = u16_to_vr(val)

    res2_sb = 1
    res2_vr = u16_to_vr(val)

    res3_sb = 2
    res3_vr = u16_to_vr(val)

    res4_sb = 3
    res4_vr = u16_to_vr(val)

    cpy_imm_16_4x(res1_sb, res2_sb, res3_sb, res4_sb, val)
    assert np.array_equal(res1_vr, convert_to_u16(diri.SB[res1_sb]))
    assert np.array_equal(res2_vr, convert_to_u16(diri.SB[res2_sb]))
    assert np.array_equal(res3_vr, convert_to_u16(diri.SB[res3_sb]))
    assert np.array_equal(res4_vr, convert_to_u16(diri.SB[res4_sb]))
    return res1_sb


#   _____         _
#  |_   _|_ _ _ _| |_ __ _ _ _
#    | |/ _` | '_|  _/ _` | ' \
#    |_|\__,_|_|  \__\__,_|_||_|


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_tartan_assignment_all_sections(d: DIRI) -> int:
    dims = {'plats': 32, 'sections': 16}

    lvr = 3
    ls = 0xFFFF   # all sections of the receiver, lvr
    tvr = 1       # temporary VR, manually allocated
    mrk = 2       # marker VR
    ms = 2        # section in mrk containing markers
    val = 0x003B  # value to write to marked plats
    donor = 4     # destination for donor matrix
                  #   (see commentary in src/belex/library.py)

    # Check known, random contents of lvr at init time.

    actual_lvr = d.glass(lvr, **dims).split('\n')
    expected_lvr = [
        '0001''1111''1101''0100''1011''0111''0011''0001',
        '1101''0110''0010''1101''1111''1011''1101''0111',
        '1001''0111''0001''1100''0111''1110''0000''1101',
        '1001''1110''1110''0101''1011''1000''1110''0101',
        '0000''1101''0000''1101''0010''0111''0100''1110',
        '0111''0110''0010''1001''1110''1111''0100''0010',
        '0111''0000''1001''0011''0011''0010''1000''0001',
        '0100''0000''1001''0111''0110''1001''0000''1110',
        '0101''1110''1110''1000''1111''0001''1111''1000',
        '1100''0000''0101''1111''0101''0110''0011''0110',
        '0101''1110''1010''1100''1100''1111''1001''1011',
        '1111''1110''1110''1010''0010''1000''1100''0111',
        '1010''1011''0010''0010''1100''1011''1001''0001',
        '1000''0001''0111''1100''0100''0110''1011''1111',
        '1010''1000''0100''1111''1101''0111''0001''0001',
        '0110''0101''1011''1011''1001''1111''0011''1010',
        ]
    assert actual_lvr == expected_lvr

    # Write a known pattern of markers.

    _write_test_markers(tvr, ms, mrk)

    # Check that expected marks are written to mrk[ms].

    actual_glass = d.glass(mrk, sections=7, plats=10)
    expected_glass = \
        '0000000000\n' \
        '0000000000\n' \
        '0001011100\n' \
        '0000000000\n' \
        '0000000000\n' \
        '0000000000\n' \
        '0000000000'
    assert actual_glass == expected_glass

    # Check that the writing of markers did not disturb lvr.

    actual_lvr = d.glass(lvr, **dims).split('\n')
    assert actual_lvr == expected_lvr

    # Write and check the donor matrix.

    tartan_imm_donor(donor, val)

    actual_donor = d.glass(donor, **dims).split('\n')
    expected_donor = [
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        ]
    assert actual_donor == expected_donor

    # Check that 'val' is written to all sections of marked plats.

    tartan_assign(lvr, ls, mrk, ms, donor, tvr)

    actual_lvr = d.glass(lvr, **dims).split('\n')
    expected_lvr = [
        '0001''1111''1101''0100''1011''0111''0011''0001',
        '1101''0111''0010''1101''1111''1011''1101''0111',
        '1000''0000''0001''1100''0111''1110''0000''1101',
        '1001''1111''1110''0101''1011''1000''1110''0101',
        '0001''1111''0000''1101''0010''0111''0100''1110',
        '0111''0111''0010''1001''1110''1111''0100''0010',
        '0110''0000''1001''0011''0011''0010''1000''0001',
        '0100''0000''1001''0111''0110''1001''0000''1110',
        '0100''1000''1110''1000''1111''0001''1111''1000',
        '1100''0000''0101''1111''0101''0110''0011''0110',
        '0100''1000''1010''1100''1100''1111''1001''1011',
        '1110''1000''1110''1010''0010''1000''1100''0111',
        '1010''1000''0010''0010''1100''1011''1001''0001',
        '1000''0000''0111''1100''0100''0110''1011''1111',
        '1010''1000''0100''1111''1101''0111''0001''0001',
        '0110''0000''1011''1011''1001''1111''0011''1010',
        ]
    assert actual_lvr == expected_lvr

    return lvr


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_tartan_assignment_some_sections(d: DIRI) -> int:
    dims = {'plats': 32, 'sections': 16}

    lvr = 3
    ls = 0x0F0F   # certain sections of the receiver, lvr
    tvr = 1       # temporary VR, manually allocated
    mrk = 2       # marker VR
    ms = 2        # section in mrk containing markers
    val = 0x003B  # value to write to marked plats
    donor = 4     # destination for donor matrix
                  #   (see commentary in src/belex/library.py)

    # Check known, random contents of lvr at init time.

    actual_lvr = d.glass(lvr, **dims).split('\n')
    expected_lvr = [
		'0001''1111''1101''0100''1011''0111''0011''0001',
		'1101''0110''0010''1101''1111''1011''1101''0111',
		'1001''0111''0001''1100''0111''1110''0000''1101',
		'1001''1110''1110''0101''1011''1000''1110''0101',
		'0000''1101''0000''1101''0010''0111''0100''1110',
		'0111''0110''0010''1001''1110''1111''0100''0010',
		'0111''0000''1001''0011''0011''0010''1000''0001',
		'0100''0000''1001''0111''0110''1001''0000''1110',
		'0101''1110''1110''1000''1111''0001''1111''1000',
		'1100''0000''0101''1111''0101''0110''0011''0110',
		'0101''1110''1010''1100''1100''1111''1001''1011',
		'1111''1110''1110''1010''0010''1000''1100''0111',
		'1010''1011''0010''0010''1100''1011''1001''0001',
		'1000''0001''0111''1100''0100''0110''1011''1111',
		'1010''1000''0100''1111''1101''0111''0001''0001',
		'0110''0101''1011''1011''1001''1111''0011''1010',
        ]
    assert actual_lvr == expected_lvr

    # Write a known pattern of markers.

    _write_test_markers(tvr, ms, mrk)

    # Check that expected marks are written to mrk[ms].

    actual_glass = d.glass(mrk, sections=7, plats=10)
    expected_glass = \
        '0000000000\n' \
        '0000000000\n' \
        '0001011100\n' \
        '0000000000\n' \
        '0000000000\n' \
        '0000000000\n' \
        '0000000000'
    assert actual_glass == expected_glass

    # Check that the writing of markers did not disturb lvr.

    actual_lvr = d.glass(lvr, **dims).split('\n')
    assert actual_lvr == expected_lvr

    # Write marker bits to donor and check.

    tartan_imm_donor(donor, val)

    actual_donor = d.glass(donor, **dims).split('\n')
    expected_donor = [
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        ]
    assert actual_donor == expected_donor

    # Check that 'val' is written to specified sections of marked plats.

    tartan_assign(lvr, ls, mrk, ms, donor, tvr)

    actual_lvr = d.glass(lvr, **dims).split('\n')
    expected_lvr = [
		'0001''1111''1101''0100''1011''0111''0011''0001',
		'1101''0111''0010''1101''1111''1011''1101''0111',
		'1000''0000''0001''1100''0111''1110''0000''1101',
		'1001''1111''1110''0101''1011''1000''1110''0101',
		'0000''1101''0000''1101''0010''0111''0100''1110',
		'0111''0110''0010''1001''1110''1111''0100''0010',
		'0111''0000''1001''0011''0011''0010''1000''0001',
		'0100''0000''1001''0111''0110''1001''0000''1110',
		'0100''1000''1110''1000''1111''0001''1111''1000',
		'1100''0000''0101''1111''0101''0110''0011''0110',
		'0100''1000''1010''1100''1100''1111''1001''1011',
		'1110''1000''1110''1010''0010''1000''1100''0111',
		'1010''1011''0010''0010''1100''1011''1001''0001',
		'1000''0001''0111''1100''0100''0110''1011''1111',
		'1010''1000''0100''1111''1101''0111''0001''0001',
		'0110''0101''1011''1011''1001''1111''0011''1010',
        ]
    assert actual_lvr == expected_lvr

    return lvr


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_tartan_xor_equals_some_sections(d: DIRI) -> int:
    dims = {'plats': 32, 'sections': 16}

    lvr = 3
    ls = 0x0F0F   # certain sections of the receiver, lvr
    tvr = 1       # temporary VR, manually allocated
    mrk = 2       # marker VR
    ms = 2        # section in mrk containing markers
    val = 0x003B  # value to write to marked plats
    donor = 4     # destination for donor matrix
                  #   (see commentary in src/belex/library.py)

    # Check known, random contents of lvr at init time.

    actual_lvr = d.glass(lvr, **dims).split('\n')
    expected_lvr = [
		'0001''1111''1101''0100''1011''0111''0011''0001',
		'1101''0110''0010''1101''1111''1011''1101''0111',
		'1001''0111''0001''1100''0111''1110''0000''1101',
		'1001''1110''1110''0101''1011''1000''1110''0101',
		'0000''1101''0000''1101''0010''0111''0100''1110',
		'0111''0110''0010''1001''1110''1111''0100''0010',
		'0111''0000''1001''0011''0011''0010''1000''0001',
		'0100''0000''1001''0111''0110''1001''0000''1110',
		'0101''1110''1110''1000''1111''0001''1111''1000',
		'1100''0000''0101''1111''0101''0110''0011''0110',
		'0101''1110''1010''1100''1100''1111''1001''1011',
		'1111''1110''1110''1010''0010''1000''1100''0111',
		'1010''1011''0010''0010''1100''1011''1001''0001',
		'1000''0001''0111''1100''0100''0110''1011''1111',
		'1010''1000''0100''1111''1101''0111''0001''0001',
		'0110''0101''1011''1011''1001''1111''0011''1010',
        ]
    assert actual_lvr == expected_lvr

    # Write a known pattern of markers.

    _write_test_markers(tvr, ms, mrk)

    # Check that expected marks are written to mrk[ms].

    actual_glass = d.glass(mrk, sections=7, plats=10)
    expected_glass = \
        '0000000000\n' \
        '0000000000\n' \
        '0001011100\n' \
        '0000000000\n' \
        '0000000000\n' \
        '0000000000\n' \
        '0000000000'
    assert actual_glass == expected_glass

    # Check that the writing of markers did not disturb lvr.

    actual_lvr = d.glass(lvr, **dims).split('\n')
    assert actual_lvr == expected_lvr

    # Write marker bits to donor and check.

    tartan_imm_donor(donor, val)

    actual_donor = d.glass(donor, **dims).split('\n')
    expected_donor = [
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        ]
    assert actual_donor == expected_donor

    # Check that 'val' is written to specified sections of marked plats.

    tartan_xor_equals(lvr, ls, mrk, ms, donor, tvr)

    actual_lvr = d.glass(lvr, **dims).split('\n')
    expected_lvr = [
		'0000''1000''1101''0100''1011''0111''0011''0001',
		'1100''0001''0010''1101''1111''1011''1101''0111',
		'1001''0111''0001''1100''0111''1110''0000''1101',
		'1000''1001''1110''0101''1011''1000''1110''0101',
		'0000''1101''0000''1101''0010''0111''0100''1110',
		'0111''0110''0010''1001''1110''1111''0100''0010',
		'0111''0000''1001''0011''0011''0010''1000''0001',
		'0100''0000''1001''0111''0110''1001''0000''1110',
		'0101''1110''1110''1000''1111''0001''1111''1000',
		'1100''0000''0101''1111''0101''0110''0011''0110',
		'0101''1110''1010''1100''1100''1111''1001''1011',
		'1111''1110''1110''1010''0010''1000''1100''0111',
		'1010''1011''0010''0010''1100''1011''1001''0001',
		'1000''0001''0111''1100''0100''0110''1011''1111',
		'1010''1000''0100''1111''1101''0111''0001''0001',
		'0110''0101''1011''1011''1001''1111''0011''1010',
        ]
    assert actual_lvr == expected_lvr

    return lvr


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_tartan_or_equals_some_sections(d: DIRI) -> int:
    dims = {'plats': 32, 'sections': 16}

    lvr = 3
    ls = 0x0F0F   # certain sections of the receiver, lvr
    tvr = 1       # temporary VR, manually allocated
    mrk = 2       # marker VR
    ms = 2        # section in mrk containing markers
    val = 0x003B  # value to write to marked plats
    donor = 4     # destination for donor matrix
                  #   (see commentary in src/belex/library.py)

    # Check known, random contents of lvr at init time.

    actual_lvr = d.glass(lvr, **dims).split('\n')
    expected_lvr = [
		'0001''1111''1101''0100''1011''0111''0011''0001',
		'1101''0110''0010''1101''1111''1011''1101''0111',
		'1001''0111''0001''1100''0111''1110''0000''1101',
		'1001''1110''1110''0101''1011''1000''1110''0101',
		'0000''1101''0000''1101''0010''0111''0100''1110',
		'0111''0110''0010''1001''1110''1111''0100''0010',
		'0111''0000''1001''0011''0011''0010''1000''0001',
		'0100''0000''1001''0111''0110''1001''0000''1110',
		'0101''1110''1110''1000''1111''0001''1111''1000',
		'1100''0000''0101''1111''0101''0110''0011''0110',
		'0101''1110''1010''1100''1100''1111''1001''1011',
		'1111''1110''1110''1010''0010''1000''1100''0111',
		'1010''1011''0010''0010''1100''1011''1001''0001',
		'1000''0001''0111''1100''0100''0110''1011''1111',
		'1010''1000''0100''1111''1101''0111''0001''0001',
		'0110''0101''1011''1011''1001''1111''0011''1010',
        ]
    assert actual_lvr == expected_lvr

    # Write a known pattern of markers.

    _write_test_markers(tvr, ms, mrk)

    # Check that expected marks are written to mrk[ms].

    actual_glass = d.glass(mrk, sections=7, plats=10)
    expected_glass = \
        '0000000000\n' \
        '0000000000\n' \
        '0001011100\n' \
        '0000000000\n' \
        '0000000000\n' \
        '0000000000\n' \
        '0000000000'
    assert actual_glass == expected_glass

    # Check that the writing of markers did not disturb lvr.

    actual_lvr = d.glass(lvr, **dims).split('\n')
    assert actual_lvr == expected_lvr

    # Write marker bits to donor and check.

    tartan_imm_donor(donor, val)

    actual_donor = d.glass(donor, **dims).split('\n')
    expected_donor = [
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        ]
    assert actual_donor == expected_donor

    # Check that 'val' is written to specified sections of marked plats.

    tartan_or_equals(lvr, ls, mrk, ms, donor, tvr)

    actual_lvr = d.glass(lvr, **dims).split('\n')
    expected_lvr = [
		'0001''1111''1101''0100''1011''0111''0011''0001',
		'1101''0111''0010''1101''1111''1011''1101''0111',
		'1001''0111''0001''1100''0111''1110''0000''1101',
		'1001''1111''1110''0101''1011''1000''1110''0101',
		'0000''1101''0000''1101''0010''0111''0100''1110',
		'0111''0110''0010''1001''1110''1111''0100''0010',
		'0111''0000''1001''0011''0011''0010''1000''0001',
		'0100''0000''1001''0111''0110''1001''0000''1110',
		'0101''1110''1110''1000''1111''0001''1111''1000',
		'1100''0000''0101''1111''0101''0110''0011''0110',
		'0101''1110''1010''1100''1100''1111''1001''1011',
		'1111''1110''1110''1010''0010''1000''1100''0111',
		'1010''1011''0010''0010''1100''1011''1001''0001',
		'1000''0001''0111''1100''0100''0110''1011''1111',
		'1010''1000''0100''1111''1101''0111''0001''0001',
		'0110''0101''1011''1011''1001''1111''0011''1010',
        ]
    assert actual_lvr == expected_lvr

    return lvr


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_tartan_and_equals_some_sections(d: DIRI) -> int:
    dims = {'plats': 32, 'sections': 16}

    lvr = 3
    ls = 0x0F0F   # certain sections of the receiver, lvr
    tvr = 1       # temporary VR, manually allocated
    mrk = 2       # marker VR
    ms = 2        # section in mrk containing markers
    val = 0x003B  # value to write to marked plats
    donor = 4     # destination for donor matrix
                  #   (see commentary in src/belex/library.py)

    # Check known, random contents of lvr at init time.

    actual_lvr = d.glass(lvr, **dims).split('\n')
    expected_lvr = [
		'0001''1111''1101''0100''1011''0111''0011''0001',
		'1101''0110''0010''1101''1111''1011''1101''0111',
		'1001''0111''0001''1100''0111''1110''0000''1101',
		'1001''1110''1110''0101''1011''1000''1110''0101',
		'0000''1101''0000''1101''0010''0111''0100''1110',
		'0111''0110''0010''1001''1110''1111''0100''0010',
		'0111''0000''1001''0011''0011''0010''1000''0001',
		'0100''0000''1001''0111''0110''1001''0000''1110',
		'0101''1110''1110''1000''1111''0001''1111''1000',
		'1100''0000''0101''1111''0101''0110''0011''0110',
		'0101''1110''1010''1100''1100''1111''1001''1011',
		'1111''1110''1110''1010''0010''1000''1100''0111',
		'1010''1011''0010''0010''1100''1011''1001''0001',
		'1000''0001''0111''1100''0100''0110''1011''1111',
		'1010''1000''0100''1111''1101''0111''0001''0001',
		'0110''0101''1011''1011''1001''1111''0011''1010',
        ]
    assert actual_lvr == expected_lvr

    # Write a known pattern of markers.

    _write_test_markers(tvr, ms, mrk)

    # Check that expected marks are written to mrk[ms].

    actual_glass = d.glass(mrk, sections=7, plats=10)
    expected_glass = \
        '0000000000\n' \
        '0000000000\n' \
        '0001011100\n' \
        '0000000000\n' \
        '0000000000\n' \
        '0000000000\n' \
        '0000000000'
    assert actual_glass == expected_glass

    # Check that the writing of markers did not disturb lvr.

    actual_lvr = d.glass(lvr, **dims).split('\n')
    assert actual_lvr == expected_lvr

    # Write marker bits to donor and check.

    tartan_imm_donor(donor, val)

    actual_donor = d.glass(donor, **dims).split('\n')
    expected_donor = [
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '1111''1111''1111''1111''1111''1111''1111''1111',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        ]
    assert actual_donor == expected_donor

    # Check that 'val' is written to specified sections of marked plats.

    tartan_and_equals(lvr, ls, mrk, ms, donor, tvr)

    actual_tvr = d.glass(tvr, **dims).split('\n')
    expected_tvr = [
        '0001''0111''0000''0000''0000''0000''0000''0000',
        '0001''0111''0000''0000''0000''0000''0000''0000',
        '0001''0111''0000''0000''0000''0000''0000''0000',
        '0001''0111''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0001''0111''0000''0000''0000''0000''0000''0000',
        '0001''0111''0000''0000''0000''0000''0000''0000',
        '0001''0111''0000''0000''0000''0000''0000''0000',
        '0001''0111''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        '0000''0000''0000''0000''0000''0000''0000''0000',
        ]
    assert actual_tvr == expected_tvr

    actual_lvr = d.glass(lvr, **dims).split('\n')
    expected_lvr = [
		'0001''1111''1101''0100''1011''0111''0011''0001',
		'1101''0110''0010''1101''1111''1011''1101''0111',
		'1000''0000''0001''1100''0111''1110''0000''1101',
		'1001''1110''1110''0101''1011''1000''1110''0101',
		'0000''1101''0000''1101''0010''0111''0100''1110',
		'0111''0110''0010''1001''1110''1111''0100''0010',
		'0111''0000''1001''0011''0011''0010''1000''0001',
		'0100''0000''1001''0111''0110''1001''0000''1110',
		'0100''1000''1110''1000''1111''0001''1111''1000',
		'1100''0000''0101''1111''0101''0110''0011''0110',
		'0100''1000''1010''1100''1100''1111''1001''1011',
		'1110''1000''1110''1010''0010''1000''1100''0111',
		'1010''1011''0010''0010''1100''1011''1001''0001',
		'1000''0001''0111''1100''0100''0110''1011''1111',
		'1010''1000''0100''1111''1101''0111''0001''0001',
		'0110''0101''1011''1011''1001''1111''0011''1010',
        ]
    assert actual_lvr == expected_lvr

    return lvr


# __      __   _ _           _           __  __          _          _
# \ \    / / _(_) |_ ___ ___| |_ ___ ___|  \/  |__ _ _ _| |_____ __| |
#  \ \/\/ / '_| |  _/ -_)___|  _/ _ \___| |\/| / _` | '_| / / -_) _` |
#   \_/\_/|_| |_|\__\___|    \__\___/   |_|  |_\__,_|_| |_\_\___\__,_|


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_randomized_write_to_marked(d: DIRI) -> int:
    r"""This is the old 'exercise 5'"""
    dims = {'plats': 32, 'sections': 16}

    lvr = 3
    tvr = 1
    mrk = 2
    ms = 2
    val = 0x003B

    actual_lvr = d.glass(lvr, **dims).split('\n')
    expected_lvr = [
        '0001''1111''1101''0100''1011''0111''0011''0001',
        '1101''0110''0010''1101''1111''1011''1101''0111',
        '1001''0111''0001''1100''0111''1110''0000''1101',
        '1001''1110''1110''0101''1011''1000''1110''0101',
        '0000''1101''0000''1101''0010''0111''0100''1110',
        '0111''0110''0010''1001''1110''1111''0100''0010',
        '0111''0000''1001''0011''0011''0010''1000''0001',
        '0100''0000''1001''0111''0110''1001''0000''1110',
        '0101''1110''1110''1000''1111''0001''1111''1000',
        '1100''0000''0101''1111''0101''0110''0011''0110',
        '0101''1110''1010''1100''1100''1111''1001''1011',
        '1111''1110''1110''1010''0010''1000''1100''0111',
        '1010''1011''0010''0010''1100''1011''1001''0001',
        '1000''0001''0111''1100''0100''0110''1011''1111',
        '1010''1000''0100''1111''1101''0111''0001''0001',
        '0110''0101''1011''1011''1001''1111''0011''1010',
        ]
    assert actual_lvr == expected_lvr

    _write_test_markers(tvr, ms, mrk)

    actual_glass = d.glass(mrk, sections=7, plats=10)
    expected_glass = \
        '0000000000\n' \
        '0000000000\n' \
        '0001011100\n' \
        '0000000000\n' \
        '0000000000\n' \
        '0000000000\n' \
        '0000000000'
    assert actual_glass == expected_glass

    actual_lvr = d.glass(lvr, **dims).split('\n')
    expected_lvr = [
        '0001''1111''1101''0100''1011''0111''0011''0001',
        '1101''0110''0010''1101''1111''1011''1101''0111',
        '1001''0111''0001''1100''0111''1110''0000''1101',
        '1001''1110''1110''0101''1011''1000''1110''0101',
        '0000''1101''0000''1101''0010''0111''0100''1110',
        '0111''0110''0010''1001''1110''1111''0100''0010',
        '0111''0000''1001''0011''0011''0010''1000''0001',
        '0100''0000''1001''0111''0110''1001''0000''1110',
        '0101''1110''1110''1000''1111''0001''1111''1000',
        '1100''0000''0101''1111''0101''0110''0011''0110',
        '0101''1110''1010''1100''1100''1111''1001''1011',
        '1111''1110''1110''1010''0010''1000''1100''0111',
        '1010''1011''0010''0010''1100''1011''1001''0001',
        '1000''0001''0111''1100''0100''0110''1011''1111',
        '1010''1000''0100''1111''1101''0111''0001''0001',
        '0110''0101''1011''1011''1001''1111''0011''1010',
        ]
    assert actual_lvr == expected_lvr

    write_to_marked(lvr, mrk, ms, val)

    actual_lvr = d.glass(lvr, **dims).split('\n')
    expected_lvr = [
        '0001''1111''1101''0100''1011''0111''0011''0001',
        '1101''0111''0010''1101''1111''1011''1101''0111',
        '1000''0000''0001''1100''0111''1110''0000''1101',
        '1001''1111''1110''0101''1011''1000''1110''0101',
        '0001''1111''0000''1101''0010''0111''0100''1110',
        '0111''0111''0010''1001''1110''1111''0100''0010',
        '0110''0000''1001''0011''0011''0010''1000''0001',
        '0100''0000''1001''0111''0110''1001''0000''1110',
        '0100''1000''1110''1000''1111''0001''1111''1000',
        '1100''0000''0101''1111''0101''0110''0011''0110',
        '0100''1000''1010''1100''1100''1111''1001''1011',
        '1110''1000''1110''1010''0010''1000''1100''0111',
        '1010''1000''0010''0010''1100''1011''1001''0001',
        '1000''0000''0111''1100''0100''0110''1011''1111',
        '1010''1000''0100''1111''1101''0111''0001''0001',
        '0110''0000''1011''1011''1001''1111''0011''1010',
        ]
    assert actual_lvr == expected_lvr

    return lvr


@parameterized_belex_test
def test_write_to_marked(d: DIRI) -> int:

    dst  = 3
    mrk  = 2
    mrks = 2
    temp = 1
    val  = 0x003B

    _write_test_markers(temp, mrks, mrk)

    actual_glass = d.glass(mrk, sections=7, plats=10)
    expected_glass = \
        '0000000000\n' \
        '0000000000\n' \
        '0001011100\n' \
        '0000000000\n' \
        '0000000000\n' \
        '0000000000\n' \
        '0000000000'
    assert actual_glass == expected_glass

    write_to_marked(dst, mrk, mrks, val)

    actual_glass = d.glass(dst, sections=7, plats=10)
    expected_glass = \
        '0001011100\n' \
        '0001011100\n' \
        '0000000000\n' \
        '0001011100\n' \
        '0001011100\n' \
        '0001011100\n' \
        '0000000000'
    assert actual_glass == expected_glass

    return dst


# __   ____  __ ___     _    _
# \ \ / /  \/  | _ \   | |  / |
#  \ V /| |\/| |   /_  | |__| |
#   \_/ |_|  |_|_|_( ) |____|_|
#                  |/


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_store_load_16(diri: DIRI) -> int:
    vr_src = 0
    vmr_l1 = 0
    expected_value = convert_to_u16(diri.hb[vr_src])
    store_16(vmr_l1, vr_src)
    cpy_imm_16(vr_src, 0)
    assert all(convert_to_u16(diri.hb[vr_src]) == 0x0000)
    load_16(vr_src, vmr_l1)
    actual_value = convert_to_u16(diri.hb[vr_src])
    assert np.array_equal(expected_value, actual_value)
    return vr_src


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_swap_vr_vmr_16(diri: DIRI) -> int:
    vr_lvr = 0
    vr_l2vr = 1
    vr_rvr = 2
    vr_r2vr = 3
    vmr_l1 = 0

    assert not np.array_equal(diri.hb[vr_lvr], diri.hb[vr_l2vr])
    assert not np.array_equal(diri.hb[vr_lvr], diri.hb[vr_rvr])
    assert not np.array_equal(diri.hb[vr_rvr], diri.hb[vr_r2vr])

    store_16(vmr_l1, vr_lvr)
    cpy_vr(vr_l2vr, vr_rvr)

    swap_vr_vmr_16(vr_l2vr, vmr_l1)
    xor_16(vr_lvr, vr_lvr, vr_l2vr)

    load_16(vr_r2vr, vmr_l1)
    xor_16(vr_rvr, vr_rvr, vr_r2vr)

    glass_actual_vr_lvr_before = diri.glass(vr_lvr, sections=16, plats=32)
    glass_actual_vr_rvr_before = diri.glass(vr_rvr, sections=16, plats=32)

    or_16(vr_lvr, vr_lvr, vr_rvr)

    glass_actual_vr_lvr_after = diri.glass(vr_lvr, sections=16, plats=32)
    assert all(convert_to_u16(diri.hb[vr_lvr]) == 0x0000)

    return vr_lvr


#  _              _             __  __ _               _____       _
# | |   __ _ _ _ (_)_ _  __ _  |  \/  (_)__ _ _ ___ __|_   _|__ __| |_ ___
# | |__/ _` | ' \| | ' \/ _` | | |\/| | / _| '_/ _ \___|| |/ -_|_-<  _(_-<
# |____\__,_|_||_|_|_||_\__, | |_|  |_|_\__|_| \___/    |_|\___/__/\__/__/
#                       |___/


# "Laning" refers to putting compatible commands in separate lanes
# of the chip. The chip has four lanes. Thus, up to four commands
# may execute at once under certain circumstances. There are two
# kinds of "compatible" commands: non-interfering and special
# cases. Non-interfering commands have disjoint SBs or disjoint
# sections in the same SB. There are also at least two special
# cases: Read Before Broadcast and Write Before Read. A way to
# remember them is that "Read" and "Broadcast" are not in
# alphabetical order. Write the rule as "Broadcast After Read" and
# "Broadcast" and "Read" are in alphabetical order. Likewise for
# Write Before Read. The following table summarizes:

# Write     *B*efore Read:      non-alpha*B*etical order
# Read      *B*efore Broadcast: non-alpha*B*etical order

# Broadcast *A*fter  Read:      *A*lphabetical order
# Read      *A*fter  Write:     *A*lphabetical order

# The following comments categorize commands into Writes, Reads, and
# Broadcasts:

#   ("Write Logic" means "SB on left-hand-side of assignment")

#   Let <SRC> be one of (INV_)?[GL, GGL, RSP16, RL, [NEWS]RL]
#   NOTA BENE: <SRC> does NOT include SB!

#   msk: is a section mask

#   As many as three VRs may be written in one clock on the
#   left-hand side of Write Logic

#   msk: SB[x]       = <SRC>, e.g., SB[9] = RL
#   msk: SB[x, y]    = <SRC>, e.g., SB[3, 14] = GL
#   msk: SB[x, y, z] = <SRC>, e.g., SB[1, 2, 3] = WRL

#   where x, y, z are each one of RN_REG_0 .. RN_REG_15.

#   SB[x] is shorthand for SB[x, x, x],
#   SB[x, y] is shorthand for SB[x, y, y]


#   ("Read Logic" means "RL on left-hand side of assignment")

#   Let <SRC> be one of (INV_)?[GL, GGL, RSP16, RL, [NEWS]RL]
#                   NOTA BENE: <SRC> does NOT include SB!

#   Let <SB> be one of SB[x], SB[x, y], SB[x, y, z], where
#   x, y, and z are numbers between 0 and 23 inclusive both ends

#     1.  msk: RL  = 0                 # read immediate
#     2.  msk: RL  = 1                 # read immediate

#     3.  msk: RL  =  <SB>
#     4.  msk: RL  =  <SRC>
#     5.  msk: RL  =  <SB> &  <SRC>

#    10.  msk: RL |=  <SB>
#    11.  msk: RL |=  <SRC>
#    12.  msk: RL |=  <SB> &  <SRC>

#    13.  msk: RL &=  <SB>
#    14.  msk: RL &=  <SRC>
#    15.  msk: RL &=  <SB> &  <SRC>

#    18.  msk: RL ^=  <SB>
#    19.  msk: RL ^=  <SRC>
#    20.  msk: RL ^=  <SB> &  <SRC>

# ----    special cases    ----

#     6.  msk: RL  =  <SB> |  <SRC>
#     7.  msk: RL  =  <SB> ^  <SRC>

#     8.  msk: RL  = ~<SB> &  <SRC>
#     9.  msk: RL  =  <SB> & ~<SRC>

#    16.  msk: RL &= ~<SB>
#    17.  msk: RL &= ~<SRC>

#    (There are three "Broadcast" commands)
#    msk: GL = RL
#    msk: GGL = RL
#    msk: RSP16 = RL


# +-+-+-+-+ +-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+
# |R|E|A|D| |B|E|F|O|R|E| |B|R|O|A|D|C|A|S|T|
# +-+-+-+-+ +-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+


# In the next test-set, we'll compare the results of
# read-before-broadcast, laned, against the results of read-before
# broadcast, not laned. Write some random data to all sections of
# RL, then read some of the sections into GL. The sections will be
# ANDed on their way to GL. We'll check that the results are the
# same in the two cases. In the bodies of the functions-under-test
# (the FUTs) are some suggestions for small modifications that
# make the tests fail.


@belex_apl
def read_before_broadcast_laned(Belex, dst: VR, sb_source: VR):
    r"""Return some randoms in 'dst', the same randoms in every
    section because the values are cycled through GL."""
    # Swap the commands in the following instruction and see the
    # test PASS. It's a really bad idea to present commands in an
    # order contrary to the order of their side effects, even
    # though they run in parallel, because a reader of your code
    # will be deceived. So put the commands back in the original
    # order, read before broadcast, when you're done seeing the
    # test pass with a humanly incorrect order.
    with apl_commands("read_before_broadcast_laned"):
        RL["0xFFFF"] <= sb_source()
        GL["048C"]   <= RL()
    with apl_commands("read_before_broadcast_laned: final output"):
        dst["0xFFFF"] <= GL()
    # don't return anything


@belex_apl
def read_before_broadcast_not_laned(Belex, dst: VR, sb_source: VR):
    r"""Return some randoms in 'dst', the same randoms in every
    section because the values are cycled through GL. Compare the
    results from this FUT (function-under-test) to the results
    from its partner FUT, 'read_before_broadcast_laned.'"""
    # Swap the next two instructions ('with' blocks of commands)
    # to see the test fail, proving that the Read must execute
    # before the Broadcast.
    with apl_commands("read_before_broadcast: first unlaned command"):
        RL["0xFFFF"] <= sb_source()
    with apl_commands("read_before_broadcast: second unlaned command"):
        GL["048C"]   <= RL()
    # Swap the prior two instructions ('with' blocks of commands)
    # to see the test fail.
    with apl_commands("read_before_broadcast unlaned: final output"):
        dst["0xFFFF"] <= GL()
    # don't return anything


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_read_before_broadcast_laning_example(diri: DIRI) -> int:
    sb0 = 0
    sb1 = 1
    sb2 = 2

    # C-sim will add to its examples any SBs whose values
    # are changed while this function executes in DIRI.

    read_before_broadcast_not_laned(sb1, sb0)
    read_before_broadcast_laned(sb2, sb0)
    xor_16(sb0, sb1, sb2)

    # How does C-sim know where to find the actual value to check
    # against the expected value? The first out-parameter of the
    # last FUT called in this test routine is treated as the VR
    # that contains the ACTUAL value. That VR is 'sb0', here. It's
    # an out-parameter, by convention, because it's in the first
    # slot of 'xor_16', and the VRs named in the first slots of
    # BELEX library calls are out parameters (they can also be
    # in-parameters).

    # How does C-sim know what the EXPECTED output is? If you
    # return a DIRI SB from this test routine, C-sim treats it as
    # the expected value. If you do not return a DIRI SB from this
    # test routine, C-sim will treat the first changed SB from
    # 0..23, filtered by used_sbs, as the expected value.

    # Check DIRI
    diri_u16s = convert_to_u16(diri.hb[sb0])
    assert all(diri_u16s == 0)

    # Check C-sim
    return sb0


# +-+-+-+-+-+ +-+-+-+-+-+-+ +-+-+-+-+
# |W|R|I|T|E| |B|E|F|O|R|E| |R|E|A|D|
# +-+-+-+-+-+ +-+-+-+-+-+-+ +-+-+-+-+


@belex_apl
def write_before_read_2_sbs_laned(
        Belex,
        dst2: VR,  # This is the output to check in C-sim
        dst1: VR,
        sb_source1: VR,
        sb_source2: VR):

    with apl_commands():
        RL["0xFFFF"] <= sb_source1()

    # To refute this test, i.e., to show that read (RL = ...) does
    # NOT happen before write (SB = ...), remove the 'with' and
    # reverse the order of commands. In that case, dst1 will be
    # overwritten with source2 and the test will fail.

    # Order of presentation is
    with apl_commands("write_before_read laned: dst1"):
        # Put RL's random junk in 'dst1'.
        # A Write command: SB on the left-hand side of assignment:
        dst1["0xFFFF"] <= RL()
        # Put new random junk in 'dst2'
        # A Read command: RL on the left-hand side of assignment:
        RL["0xFFFF"]   <= sb_source2()

    with apl_commands("write_before_read laned: dst2"):
        # Put RL's new random junk in 'dst2'.
        dst2["0xFFFF"] <= RL()

    # Check in test fn: dst2 (out param) should == source2.
    # Don't return anything.


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_write_before_read_2_sbs_laning_example(diri: DIRI) -> int:
    sb_source1 = 1
    sb_source2 = 2
    dst1  = 3
    dst2  = 5

    dst_ult = 6
    dst_ult2 = 7

    # This test does dst2 := source2, dst1 := source1, using
    # write-before-read parallelism in different SBs, no matter
    # about the groups 0..7, 8..15, 16..24, and overlapping
    # section masks (0xFFFF).

    write_before_read_2_sbs_laned(
        dst2, dst1, sb_source1, sb_source2)
    # Compare sb_source2, dst2 (their XOR must be zero)
    xor_16(dst_ult, sb_source2, dst2)

    # Check so in DIRI:
    diri_u16s = convert_to_u16(diri.hb[dst_ult])
    assert all(diri_u16s == 0)

    # Check dst1 and dst2 are (statistically) different:
    xor_16(dst_ult2, dst1, dst2)
    diri_u16s = convert_to_u16(diri.hb[dst_ult2])
    assert not all(diri_u16s == 0)

    # Check C-sim actual dst2 (first out parameter in
    # write_before_read_2_sbs_laned) against this expected:
    return dst2

    # End of test_write_before_read_2_sbs_laning_example


@belex_apl
def write_before_read_laned(Belex, dst1: VR, dst2: VR, sb_source: VR):
    r"""Return some randoms inside dst1, Return zeros inside dst2.
    Inside a single instruction, copy the randoms out of RL and
    also clear RL. If the write (dst1 = ...) is /not/ done before
    the read (RL = 0), then the output would be zero and not random."""

    with apl_commands():
        RL["0xFFFF"] <= sb_source()  # randoms from sb_source

    # Order of presentation does NOT matter for the two commands
    # (write before read) in the instruction below. However, it is
    # VERY bad practice to present the read (RL = ...) before the
    # write (SB = ...), because a human reader will assume the
    # commands are executed in the order presented, and they are
    # NOT in this case. Both DIRI and the C-sim will execute the
    # write before the read no matter the order of presentation.

    with apl_commands("write_before_read laned: dst1"):
        dst1["0xFFFF"] <= RL() # ... randoms copied to dst1
        RL["0xFFFF"] <= 0      # ... BEFORE clearing RL
        # but in parallel, in one clock, in one instruction.

    with apl_commands("write_before_read laned: dst2"):
        dst2["0xFFFF"] <= RL() # ... and dst2 should contain 0

    # don't return anything


@belex_apl
def write_before_read_not_laned(Belex, dst1: VR, dst2: VR, sb_source: VR):
    r"""Return some randoms inside dst1, Return zeros inside dst2.
    Inside a single instruction, copy the randoms out of RL and
    also clear RL. If the write (dst1 = ...) is /not/ done before
    the read (RL = 0), then the output would be zero and not random."""

    with apl_commands():
        RL["0xFFFF"] <= sb_source()  # randoms from sb_source

    # Swap the next two instructions ('with' blocks) to see the test fail.
    with apl_commands("write_before_read unlaned: first command"):
        dst1["0xFFFF"] <= RL() # ... copied to dst1

    with apl_commands("write_before_read unlaned: second command, dst1"):
        RL["0xFFFF"] <= 0
    # Swap the previous two instructions ('with' blocks) to see the test fail

    with apl_commands("write_before_read unlaned: dst2"):
        dst2["0xFFFF"] <= RL() # and dst2 should contain 0

    # don't return anything


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_write_before_read_dst_laning_example(diri: DIRI) -> int:
    sb_source = 2  # random bits from here
    dst1_l    = 3  # ... should be written into dst1_l (check here)
    dst2_l    = 1  # ... and this should be 0 (check in another test)

    dst_ult = 5

    # Replace the following FUT (function-under-test) with a call
    # to write_before_not_laned. Play around with the order of
    # commands in write_before_not_laned to make the test fail in
    # ways you expect. Test your understanding that way.

    write_before_read_laned(dst1_l, dst2_l, sb_source)

    # Check that sb_source == dst1_l by checking that
    # sb_source XOR dst1_l is zero.

    # Omit next line to see the test fail. In that case, the
    # actual will be dst1_l (out param of write_before_read...)
    # and the expected will be dst_ult.

    xor_16(dst_ult, sb_source, dst1_l)  # Check sb_source == dst1_l

    # BELEX tells C-sim that 'dst_ult', the out-parameter in the
    # FUT xor_16, contains the actual values to check against the
    # expected values. The expected values are in 'dst_ult', which
    # is returned from this test function.

    # Check DIRI that the expected dst_ult is zero.
    diri_u16s = convert_to_u16(diri.hb[dst_ult])
    assert all(diri_u16s == 0)

    # Check C-sim dst_ult actual against dst_ult expected
    return dst_ult  # expect 0


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_write_before_read_rl_laning_example(diri: DIRI) -> int:
    sb_source = 2  # random bits from here
    dst1_l    = 3  # ... should be written into dst1_l (check in another test)
    dst2_l    = 1  # ... and this should be 0 (check here)

    dst_ult = 5

    # Replace the following FUT (function-under-test) with a call
    # to write_before_not_laned. Play around with the order of
    # commands in write_before_not_laned to make the test fail in
    # ways you expect. Test your understanding that way.

    write_before_read_laned(dst1_l, dst2_l, sb_source)
    and_16(dst_ult, sb_source, dst2_l)  # Check dst2_l == 0

    # Check DIRI
    diri_u16s = convert_to_u16(diri.hb[dst_ult])
    assert all(diri_u16s == 0)

    # Check C-sim
    return dst_ult  # expect 0


# +-+-+-+-+ +-+-+ +-+-+-+-+-+-+-+-+
# |T|E|S|T| |S|B| |"|g|r|o|u|p|s|"|
# +-+-+-+-+ +-+-+ +-+-+-+-+-+-+-+-+


@belex_apl
def sb_groups_multi(Belex, sb: VR, sb2: VR, sb_src: VR):

    r"""Copy data from all sections of sb_src into BOTH sb and sb2. sb
    and sb2 must be in the same 'group' (yet another meaning for
    this word): both must be in [0..7] or both must be in [8..15]
    or both must be in [16..23].
    """

    fs = "0xFFFF"
    msk = Belex.Mask(fs)

    RL[fs]       <= sb_src()
    msk[sb, sb2] <= RL()

    # In test function, check that contents sb == contents sb2 ==
    # contents sb_src


def check_sbs_for_equal_contents(diri, sb1, sb2):
    res1 = convert_to_u16(diri.hb[sb1])
    res2 = convert_to_u16(diri.hb[sb2])
    assert all(res1 == res2)


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_sb_groups_multi(diri: DIRI) -> int:

    r"""SB 'groups' (yet another use of the word 'group'), are supposed
    to comprise SBs 0..7, 8..15, 16..23. Test that both SBs in a
    multi on the left-hand side of an assignment statement must be
    from the same 'group', even though the SB on the right can be
    from another 'group.'
    """

    sb =  1      # pick VRs randomly

    # Change this test and pick a second SB in a wrong group:
    sb2 = 2      # pick one in the same group (or not ...); # if
    # not, expect C-sim to reject the program with a fatal
    # synchronization error.

    sb_src = 11  # pick one in another group from sb

    sb_groups_multi(sb, sb2, sb_src)

    check_sbs_for_equal_contents(diri, sb, sb2)
    check_sbs_for_equal_contents(diri, sb2, sb_src)

    return sb  # expected values from DIRI into C-sim.



# +-+-+-+ +-+-+-+-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+ +-+-+
# |S|R|L| |C|o|m|p|a|t|i|b|i|l|i|t|y| |w|i|t|h| |R|L|
# +-+-+-+ +-+-+-+-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+ +-+-+


@belex_apl
def srl_compat(Belex, x: VR, for_srl: VR):

    r"""This is a Read (RL = ...) After Write (SB = ...) test; ensure
    that contents written in the first step are reading from the
    old RL."""

    # set up
    RL[:]  <= for_srl() # put some junk in RL
    GGL[:] <= RL()      # put some junk in GGL

    with apl_commands():

        # Write the first bit of each nibble from the junk that's
        # in the second bit of each nibble of RL. We tell by
        # looking at the file 'belex-examples.h' in directory
        # /libs-gvml/subprojects/belex/apl/test_llb_srl_compat:
        # x[plat=0] begins life as 0x632B (tell from the
        # command-line output or from DIRI in the test function
        # below). RL begins life as 'for_srl', whose first plat is
        # 0xBDD7. The SECOND bits of BDD7 are 1001. Overwriting
        # the first bits of x with 1001 changes 632B into 722B.

        x["0x1111"]  <= SRL()

        # Change the second bit of each nibble of RL, so the
        # results will not be as expected if this read (RL = ...)
        # is done out-of-order.

        RL["0x2222"] &= ( x() & GGL() )



@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_srl_compat(diri: DIRI) -> int:

    x =  1

    actual_x_plat_0 = convert_to_u16(diri.hb[x])[0]
    assert actual_x_plat_0 == 0x3453

    x_cpy = 3

    cpy_vr(x_cpy, x)

    for_srl = 2

    actual_for_srl_plat_0 = convert_to_u16(diri.hb[for_srl])[0]
    assert actual_for_srl_plat_0 == 0x320F

    dst_ult = 11

    srl_compat(x, for_srl)

    # Check x gets modified

    assert convert_to_u16(diri.hb[x])[0] == 0x3543

    # This next test (not all equal) is implied by the last
    # test, but is a good pattern and exercise to keep here.

    xor_16(dst_ult, x, x_cpy)

    assert not all(convert_to_u16(diri.hb[dst_ult]) == 0)

    return x  # expected values from DIRI into C-sim.


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# |R|E|A|D|-|R|E|A|D|-|B|R|O|A|D|C|A|S|T|
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


@belex_apl
def read_read_broadcast_laned(Belex, out: VR, sb1: VR, sb2: VR,
                              xtra: VR):
    RL[:] <= xtra()
    with apl_commands("presentation order is actual order."):
        RL["0x1111"]  <= sb2() | SRL()
        RL["0x2222"]  <= sb1()
        GGL["0xFFFF"] <= RL()
    out[:] <= GGL()


@belex_apl
def read_read_broadcast_not_laned(Belex, out: VR, sb1: VR, sb2: VR,
                                  xtra: VR):
    # Cannot reorder any of the commands below.
    RL[:] <= xtra()
    RL["0x1111"]  <= sb2() | SRL()
    RL["0x2222"]  <= sb1()
    GGL["0xFFFF"] <= RL()  # Does not work if GGL command is moved up.

    out[:] <= GGL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_read_read_broadcast(diri: DIRI) -> int:
    sb1   =  5
    sb2   =  6
    out_l =  7
    out_n =  8
    ult   =  9
    xtra  = 10

    read_read_broadcast_laned    (out_l, sb1, sb2, xtra)
    read_read_broadcast_not_laned(out_n, sb1, sb2, xtra)

    xor_16(ult, out_l, out_n)
    assert all(convert_to_u16(diri.hb[ult]) == 0)

    return ult


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# |W|R|I|T|E|-|W|R|I|T|E|-|R|E|A|D|
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


@belex_apl
def write_write_read_laned(
        Belex, out: VR, sb5: VR, sb6: VR, sb7: VR, xtra: VR):
    eights = "0x8888"
    msk8 = Belex.Mask(eights)

    eff = "0x0F00"
    mskF = Belex.Mask(eff)

    RL[:] <= xtra()

    with apl_commands():
        sb7[:]  <= RL()
        RL[eff] <= sb6()

    RL[eff] |= sb5() & sb7()

    out[:] <= RL()


@belex_apl
def write_write_read_not_laned(
        Belex, out: VR, sb5: VR, sb6: VR, sb7: VR, xtra: VR):
    # Cannot reorder any of the commands below.

    eff = "0x0F00"
    mskF = Belex.Mask(eff)

    RL[:] <= xtra()

    sb7[:]  <= RL()
    RL[eff] <= sb7() & sb5()
    RL[eff] |= sb6()

    out[:] <= RL()


@parameterized_belex_test(
    repeatably_randomize_half_bank=True)
def test_write_write_read(diri: DIRI) -> int:
    sb2   =  5
    sb3   =  6
    sb4   =  7
    out_l =  8
    out_n =  9
    ult   = 10
    xtra  = 11

    write_write_read_laned    (out_l, sb2, sb3, sb4, xtra)
    write_write_read_not_laned(out_n, sb2, sb3, sb4, xtra)

    xor_16(ult, out_l, out_n)
    assert all(convert_to_u16(diri.hb[ult]) == 0)

    return ult


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# |W|R|I|T|E|-|W|R|I|T|E|-|R|E|A|D|
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


@belex_apl
def write_read_read_laned(
        Belex,
        out: VR,
        sb1: VR, sb2: VR, sb3: VR, sb4: VR,
        xtra: VR):
    eights = "0x8888"
    msk8 = Belex.Mask(eights)

    eff = "0x000F"
    mskF = Belex.Mask(eff)

    RL[:] <= xtra()

    with apl_commands():
        msk8[sb1, sb2] <= RL()
        RL[eff] <= sb4()

    RL[eff] |= sb2() & sb3()

    out[:] <= RL()


@belex_apl
def write_read_read_not_laned(
        Belex,
        out: VR,
        sb1: VR, sb2: VR, sb3: VR, sb4: VR,
        xtra: VR):
    # Cannot reorder any of the commands below.

    eights = "0x8888"
    msk8 = Belex.Mask(eights)

    eff = "0x000F"
    mskF = Belex.Mask(eff)

    RL[:] <= xtra()

    msk8[sb1, sb2] <= RL()
    RL[eff] <= sb2() & sb3()
    RL[eff] |= sb4()

    out[:] <= RL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_write_read_read(diri: DIRI) -> int:
    sb1   =  4
    sb2   =  5
    sb3   =  6
    sb4   =  7
    out_l =  8
    out_n =  9
    ult   = 10
    xtra  = 11

    write_read_read_laned    (out_l, sb1, sb2, sb3, sb4, xtra)
    write_read_read_not_laned(out_n, sb1, sb2, sb3, sb4, xtra)

    xor_16(ult, out_l, out_n)
    assert all(convert_to_u16(diri.hb[ult]) == 0)

    return ult


# +-+-+ +-+-+-+ +-+-+-+-+-+-+
# |R|L| |A|N|D| |I|N|V|_|R|L|
# +-+-+ +-+-+-+ +-+-+-+-+-+-+


@belex_apl
def rl_and_inv_rl_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    with apl_commands():
        out["0x00FF"] <= RL()
        out["0xFF00"] <= INV_RL()


@belex_apl
def rl_and_inv_rl_not_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    out["0x00FF"] <= RL()
    out["0xFF00"] <= INV_RL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_rl_and_inv_rl(diri: DIRI) -> int:
    ult = 0
    out_laned = 1
    out_not_laned = 2
    x = 3

    rl_and_inv_rl_laned(out_laned, x)
    rl_and_inv_rl_not_laned(out_not_laned, x)

    xor_16(ult, out_laned, out_not_laned)
    assert all(convert_to_u16(diri.hb[ult]) == 0x0000)
    return ult


# +-+-+ +-+-+-+ +-+-+-+
# |R|L| |A|N|D| |N|R|L|
# +-+-+ +-+-+-+ +-+-+-+


@belex_apl
def rl_and_nrl_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    with apl_commands():
        out["0x00FF"] <= RL()
        out["0xFF00"] <= NRL()


@belex_apl
def rl_and_nrl_not_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    out["0x00FF"] <= RL()
    out["0xFF00"] <= NRL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_rl_and_nrl(diri: DIRI) -> int:
    ult = 0
    out_laned = 1
    out_not_laned = 2
    x = 3

    rl_and_nrl_laned(out_laned, x)
    rl_and_nrl_not_laned(out_not_laned, x)

    xor_16(ult, out_laned, out_not_laned)
    assert all(convert_to_u16(diri.hb[ult]) == 0x0000)
    return ult


# +-+-+ +-+-+-+ +-+-+-+-+-+-+-+
# |R|L| |A|N|D| |I|N|V|_|N|R|L|
# +-+-+ +-+-+-+ +-+-+-+-+-+-+-+


@belex_apl
def rl_and_inv_nrl_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    with apl_commands():
        out["0x00FF"] <= RL()
        out["0xFF00"] <= INV_NRL()


@belex_apl
def rl_and_inv_nrl_not_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    out["0x00FF"] <= RL()
    out["0xFF00"] <= INV_NRL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_rl_and_inv_nrl(diri: DIRI) -> int:
    ult = 0
    out_laned = 1
    out_not_laned = 2
    x = 3

    rl_and_inv_nrl_laned(out_laned, x)
    rl_and_inv_nrl_not_laned(out_not_laned, x)

    xor_16(ult, out_laned, out_not_laned)
    assert all(convert_to_u16(diri.hb[ult]) == 0x0000)
    return ult


# +-+-+ +-+-+-+ +-+-+-+-+-+
# |R|L| |A|N|D| |R|S|P|1|6|
# +-+-+ +-+-+-+ +-+-+-+-+-+


@belex_apl
def rl_and_rsp16_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    RSP16[:] <= RL()
    RSP_START_RET()
    with apl_commands():
        out["0x00FF"] <= RL()
        out["0xFF00"] <= RSP16()
    NOOP()
    NOOP()
    RSP_END()


@belex_apl
def rl_and_rsp16_not_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    RSP16[:] <= RL()
    RSP_START_RET()
    out["0x00FF"] <= RL()
    out["0xFF00"] <= RSP16()
    NOOP()
    NOOP()
    RSP_END()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_rl_and_rsp16(diri: DIRI) -> int:
    ult = 0
    out_laned = 1
    out_not_laned = 2
    x = 3

    rl_and_rsp16_laned(out_laned, x)
    rl_and_rsp16_not_laned(out_not_laned, x)

    xor_16(ult, out_laned, out_not_laned)
    assert all(convert_to_u16(diri.hb[ult]) == 0x0000)
    return ult


# +-+-+ +-+-+-+ +-+-+
# |R|L| |A|N|D| |G|L|
# +-+-+ +-+-+-+ +-+-+


@belex_apl
def rl_and_gl_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    GL[:] <= RL()
    with apl_commands():
        out["0x00FF"] <= RL()
        out["0xFF00"] <= GL()


@belex_apl
def rl_and_gl_not_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    GL[:] <= RL()
    out["0x00FF"] <= RL()
    out["0xFF00"] <= GL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_rl_and_gl(diri: DIRI) -> int:
    ult = 0
    out_laned = 1
    out_not_laned = 2
    x = 3

    rl_and_gl_laned(out_laned, x)
    rl_and_gl_not_laned(out_not_laned, x)

    xor_16(ult, out_laned, out_not_laned)
    assert all(convert_to_u16(diri.hb[ult]) == 0x0000)
    return ult


# +-+-+ +-+-+-+ +-+-+-+
# |R|L| |A|N|D| |G|G|L|
# +-+-+ +-+-+-+ +-+-+-+


@belex_apl
def rl_and_ggl_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    GGL[:] <= RL()
    with apl_commands():
        out["0x00FF"] <= RL()
        out["0xFF00"] <= GGL()


@belex_apl
def rl_and_ggl_not_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    GGL[:] <= RL()
    out["0x00FF"] <= RL()
    out["0xFF00"] <= GGL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_rl_and_ggl(diri: DIRI) -> int:
    ult = 0
    out_laned = 1
    out_not_laned = 2
    x = 3

    rl_and_ggl_laned(out_laned, x)
    rl_and_ggl_not_laned(out_not_laned, x)

    xor_16(ult, out_laned, out_not_laned)
    assert all(convert_to_u16(diri.hb[ult]) == 0x0000)
    return ult


# +-+-+-+-+-+ +-+-+-+ +-+-+-+-+-+-+-+-+-+
# |R|S|P|1|6| |A|N|D| |I|N|V|_|R|S|P|1|6|
# +-+-+-+-+-+ +-+-+-+ +-+-+-+-+-+-+-+-+-+


@belex_apl
def rsp16_and_inv_rsp16_laned(Belex,
                              out: VR, x: VR,
                              rsp16_mask: Mask):
    RL[:] <= x()
    RSP16[:] <= RL()
    RSP_START_RET()
    # It seems broadcasting to RSP is a one-way operation. If you want to
    # broadcast into RSP and then pull the data back out, you must first call
    # RSP_START_RET() to retrieve it.
    with apl_commands():
        out[rsp16_mask] <= RSP16()
        out[~rsp16_mask] <= INV_RSP16()
    NOOP()
    NOOP()
    RSP_END()


@belex_apl
def rsp16_and_inv_rsp16_not_laned(Belex,
                                  out: VR, x: VR,
                                  rsp16_mask: Mask):
    RL[:] <= x()
    RSP16[:] <= RL()
    RSP_START_RET()
    # It seems broadcasting to RSP is a one-way operation. If you want to
    # broadcast into RSP and then pull the data back out, you must first call
    # RSP_START_RET() to retrieve it.
    out[rsp16_mask] <= RSP16()
    out[~rsp16_mask] <= INV_RSP16()
    NOOP()
    NOOP()
    RSP_END()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_rsp16_and_inv_rsp16(diri: DIRI) -> int:
    ult = 0
    out_laned = 1
    out_not_laned = 2
    x = 3

    rsp16_and_inv_rsp16_laned(out_laned, x, 0x00FF)
    rsp16_and_inv_rsp16_not_laned(out_not_laned, x, 0x00FF)

    xor_16(ult, out_laned, out_not_laned)
    assert all(convert_to_u16(diri.hb[ult]) == 0x0000)
    return ult


@hypothesis.settings(max_examples=3, deadline=None)
@given(rsp16_mask=Mask_strategy())
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_random_rsp16_and_inv_rsp16(diri: DIRI, rsp16_mask: int) -> int:

    ult = 0
    out_laned = 1
    out_not_laned = 2
    x = 3

    rsp16_and_inv_rsp16_laned(out_laned, x, rsp16_mask)
    rsp16_and_inv_rsp16_not_laned(out_not_laned, x, rsp16_mask)

    xor_16(ult, out_laned, out_not_laned)
    assert all(convert_to_u16(diri.hb[ult]) == 0x0000)
    return ult


# +-+-+-+ +-+-+-+-+-+-+-+ +-+-+-+ +-+-+-+-+-+-+-+
# |R|S|P| |I|N|V|_|R|S|P| |V|I|A| |L|I|B|R|A|R|Y|
# +-+-+-+ +-+-+-+-+-+-+-+ +-+-+-+ +-+-+-+-+-+-+-+


@belex_apl
def rsp16_and_inv_rsp16_laned_with_rsp_out_in(Belex,
                                              out: VR, x: VR,
                                              rsp16_mask: Mask):
    RL[:] <= x()
    rsp_out_in("0xFFFF")
    # This is an alternative way to rsp16_and_inv_rsp16_laned to broadcast to
    # RSP and retrieve its data.
    with apl_commands():
        out[rsp16_mask] <= RSP16()
        out[~rsp16_mask] <= INV_RSP16()
    NOOP()
    NOOP()
    RSP_END()


@belex_apl
def rsp16_and_inv_rsp16_not_laned_with_rsp_out_in(Belex,
                                                  out: VR, x: VR,
                                                  rsp16_mask: Mask):
    RL[:] <= x()
    rsp_out_in("0xFFFF")
    # This is an alternative way to rsp16_and_inv_rsp16_not_laned to broadcast
    # to RSP and retrieve its data.
    out[rsp16_mask] <= RSP16()
    out[~rsp16_mask] <= INV_RSP16()
    NOOP()
    NOOP()
    RSP_END()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_rsp16_and_inv_rsp16_with_rsp_out_in(diri: DIRI) -> int:
    ult = 0
    out_laned = 1
    out_not_laned = 2
    x = 3

    rsp16_and_inv_rsp16_laned_with_rsp_out_in(out_laned, x, 0x00FF)
    rsp16_and_inv_rsp16_not_laned_with_rsp_out_in(out_not_laned, x, 0x00FF)

    xor_16(ult, out_laned, out_not_laned)
    assert all(convert_to_u16(diri.hb[ult]) == 0x0000)
    return ult


@hypothesis.settings(max_examples=3, deadline=None)
@given(rsp16_mask=Mask_strategy())
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_random_rsp16_and_inv_rsp16_with_rsp_out_in(diri: DIRI, rsp16_mask: int) -> int:

    ult = 0
    out_laned = 1
    out_not_laned = 2
    x = 3

    rsp16_and_inv_rsp16_laned_with_rsp_out_in(out_laned, x, rsp16_mask)
    rsp16_and_inv_rsp16_not_laned_with_rsp_out_in(out_not_laned, x, rsp16_mask)

    xor_16(ult, out_laned, out_not_laned)
    assert all(convert_to_u16(diri.hb[ult]) == 0x0000)
    return ult



# +-+-+ +-+-+-+ +-+-+-+-+-+
# |G|L| |A|N|D| |R|S|P|1|6|
# +-+-+ +-+-+-+ +-+-+-+-+-+


@belex_apl
def gl_and_rsp16_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    GL[:] <= RL()
    RSP16[:] <= RL()
    RSP_START_RET()
    with apl_commands():
        out["0x00FF"] <= GL()
        out["0xFF00"] <= RSP16()
    NOOP()
    NOOP()
    RSP_END()


@belex_apl
def gl_and_rsp16_not_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    GL[:] <= RL()
    RSP16[:] <= RL()
    RSP_START_RET()
    out["0x00FF"] <= GL()
    out["0xFF00"] <= RSP16()
    NOOP()
    NOOP()
    RSP_END()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_gl_and_rsp16(diri: DIRI) -> int:
    ult = 0
    out_laned = 1
    out_not_laned = 2
    x = 3

    gl_and_rsp16_laned(out_laned, x)
    gl_and_rsp16_not_laned(out_not_laned, x)

    xor_16(ult, out_laned, out_not_laned)
    assert all(convert_to_u16(diri.hb[ult]) == 0x0000)
    return ult


# +-+-+ +-+-+-+ +-+-+-+
# |G|L| |A|N|D| |G|G|L|
# +-+-+ +-+-+-+ +-+-+-+


@belex_apl
def gl_and_ggl_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    GL[:] <= RL()
    GGL[:] <= RL()
    with apl_commands():
        out["0x00FF"] <= GL()
        out["0xFF00"] <= GGL()


@belex_apl
def gl_and_ggl_not_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    GL[:] <= RL()
    GGL[:] <= RL()
    out["0x00FF"] <= GL()
    out["0xFF00"] <= GGL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_gl_and_ggl(diri: DIRI) -> int:
    ult = 0
    out_laned = 1
    out_not_laned = 2
    x = 3

    gl_and_ggl_laned(out_laned, x)
    gl_and_ggl_not_laned(out_not_laned, x)

    xor_16(ult, out_laned, out_not_laned)
    assert all(convert_to_u16(diri.hb[ult]) == 0x0000)
    return ult


# +-+-+-+ +-+-+-+ +-+-+-+-+-+
# |G|G|L| |A|N|D| |R|S|P|1|6|
# +-+-+-+ +-+-+-+ +-+-+-+-+-+


@belex_apl
def ggl_and_rsp16_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    GGL[:] <= RL()
    RSP16[:] <= RL()
    RSP_START_RET()
    with apl_commands():
        out["0x00FF"] <= GGL()
        out["0xFF00"] <= RSP16()
    NOOP()
    NOOP()
    RSP_END()


@belex_apl
def ggl_and_rsp16_not_laned(Belex, out: VR, x: VR):
    RL[:] <= x()
    GGL[:] <= RL()
    RSP16[:] <= RL()
    RSP_START_RET()
    out["0x00FF"] <= GGL()
    out["0xFF00"] <= RSP16()
    NOOP()
    NOOP()
    RSP_END()


@belex_apl
def reset_rsp16_and_ggl(Belex, out: VR) -> None:
    RL[::] <= 0
    RSP16[::] <= RL()
    GGL[::] <= RL()
    RSP_START_RET()
    out["0xFF00"] <= GGL()
    out["0x00FF"] <= RSP16()
    RSP_END()
    NOOP()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_ggl_and_rsp16(diri: DIRI) -> int:
    ult = 0
    out_laned = 1
    out_not_laned = 2
    x = 3

    reset_rsp16_and_ggl(ult)
    ggl_and_rsp16_laned(out_laned, x)
    ggl_and_rsp16_not_laned(out_not_laned, x)

    xor_16(ult, out_laned, out_not_laned)
    assert all(convert_to_u16(diri.hb[ult]) == 0x0000)
    return ult


# +-+-+-+-+-+-+-+-+-+-+ +-+-+-+ +-+-+-+-+-+-+ +-+-+-+-+ +-+-+-+-+ +-+-+-+
# |E|X|H|A|U|S|T|I|V|E| |S|R|C| |C|O|M|P|A|T| |T|E|S|T| |W|I|T|H| |A|N|D|
# +-+-+-+-+-+-+-+-+-+-+ +-+-+-+ +-+-+-+-+-+-+ +-+-+-+-+ +-+-+-+-+ +-+-+-+
# +-+-+-+-+-+-+-+ +-+-+-+-+-+-+
# |W|I|T|H|O|U|T| |L|A|N|I|N|G|
# +-+-+-+-+-+-+-+ +-+-+-+-+-+-+


@belex_apl
def src1_and_src2_laned_and_not_laned(Belex, out: VR, tmp1: VR, tmp2: VR, x: VR, msk: Mask):
    srcs = [
        RL, INV_RL,
        NRL, INV_NRL,
        ERL, INV_ERL,
        WRL, INV_WRL,
        SRL, INV_SRL,
        # FIXME: RSP16, here, causes issues in generated code:
        RSP16, INV_RSP16,
        GL, INV_GL,
        GGL, INV_GGL,
    ]

    src_inits = {
        RL: RL,
        INV_RL: RL,

        NRL: RL,
        INV_NRL: RL,

        ERL: RL,
        INV_ERL: RL,

        WRL: RL,
        INV_WRL: RL,

        SRL: RL,
        INV_SRL: RL,

        RSP16: RSP16,
        INV_RSP16: RSP16,

        GL: GL,
        INV_GL: GL,

        GGL: GGL,
        INV_GGL: GGL,
    }

    # Initialize the out VR with 0
    RL[:] <= 0
    out[:] <= RL()

    for index, src1 in enumerate(srcs):
        for src2 in srcs[1 + index:]:
            RL[:] <= x()

            with apl_commands(f"{src1.symbol} (and) {src2.symbol}"):
                src1_init = src_inits[src1]
                src1_init[:] <= RL()

                src2_init = src_inits[src2]
                if src2_init is not src1_init:
                    src2_init[:] <= RL()

            if RSP16 in [src1_init, src2_init]:
                RSP_START_RET()

            # Laned
            with apl_commands():
                tmp1[msk] <= src1()
                tmp1[~msk] <= src2()

            # Not Laned
            tmp2[msk] <= src1()
            tmp2[~msk] <= src2()

            if RSP16 in [src1_init, src2_init]:
                NOOP()
                NOOP()
                RSP_END()

            # Collect the results
            RL[:] <= tmp1()
            RL[:] ^= tmp2()
            out[:] |= RL()


@hypothesis.settings(max_examples=3, deadline=None)
@given(msk=Mask_strategy())
@parameterized_belex_test(repeatably_randomize_half_bank=True,
                          # TODO: Once we fix the (1) RSP16 and (2) >= 1025
                          # cycle bugs, enable code generation again.
                          generate_code=False)
def test_src1_and_src2_laned_and_not_laned(diri: DIRI, msk: int) -> int:

    out = 0
    tmp1 = 1
    tmp2 = 2
    x = 3

    src1_and_src2_laned_and_not_laned(out, tmp1, tmp2, x, msk)
    assert all(convert_to_u16(diri.hb[out]) == 0x0000)
    return out


# +-+-+-+-+-+-+-+-+-+-+-+ +-+-+ +-+-+-+ +-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+
# |D|E|-|M|O|R|G|A|N|'|S| |O|N| |T|H|E| |R|I|G|H|T|-|H|A|N|D| |S|I|D|E|
# +-+-+-+-+-+-+-+-+-+-+-+ +-+-+ +-+-+-+ +-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+


# @belex_apl
# def demorgans_rhs_complemented(
#         Belex,
#         out: VR,
#         sb1: VR, sb2: VR):
#     # Check that the following expression means what we think ...
#     RL[:] <= ~(sb1() & sb2()) & ~RSP16()
#     out[:] <= RL()


# @belex_apl
# def demorgans_rhs_not_complemented(
#         Belex,
#         out: VR,
#         sb1: VR, sb2: VR,
#         xtra: VR):

#     # ... and this is what we think: (~sb1() | ~sb2())
#     RL[:]   <= ~sb1()
#     xtra[:] <= RL()
#     RL[:]   <= ~sb2()
#     RL[:]   |= xtra()

#     out[:] <= RL()


# @parameterized_belex_test(
#     repeatably_randomize_half_bank=True)
# def test_demorgans_rhs(diri: DIRI) -> int:
#     sb1   =  4
#     sb2   =  5
#     out_l =  8
#     out_n =  9
#     ult   = 10
#     xtra  = 11

#     demorgans_rhs_complemented    (out_l, sb1, sb2)
#     demorgans_rhs_not_complemented(out_n, sb1, sb2, xtra)

#     xor_16(ult, out_l, out_n)
#     assert all(convert_to_u16(diri.hb[ult]) == 0)

#     return ult


# +-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+
# |L|A|N|I|N|G| |L|A|B|O|R|A|T|O|R|Y|
# +-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+


@belex_apl
def laning_laboratory_001_baseline(Belex, out: VR, sb1: VR, sb2: VR):

    sb3 = RN_REG_T3
    sb4 = RN_REG_T4
    sb5 = RN_REG_T5
    sb6 = RN_REG_T6
    sb7 = RN_REG_T0

    # Trick to zero-out RL (2 cycles)
    #     001 [0xffff : RL = SB[1];]
    #     002 [0xffff : RL ^= SB[1];]
    RL[:] <= sb1()  # 001 from original.txt
    RL[:] ^= sb1()

    #     003 [0xffff : SB[7] = RL;;0xffff : RL = SB[1,2];]
    with apl_commands("read after write, 1st instance"):
        sb7[:] <= RL()
        RL [:] <= sb1() & sb2()

    #     004 [0xffff : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write, 2nd instance"):
        sb6[:] <= RL()
        RL [:] <= sb1()

    #     005 [0xffff : RL ^= SB[2];]
    RL[:] ^= sb2()

    #     006 [0xffff : SB[3,5] = RL;]
    xffff = Belex.Mask("0xffff")
    xffff[sb3, sb5] <= RL()

    #     007 [0x2222 : RL = SB[5];;0x1111 : RL = SB[6] & SRL;]
    with apl_commands("compatible sections: 1st instance"):
        RL["0x2222"] <= sb5()
        RL["0x1111"] <= sb6() & SRL()  # uses old RL

    #     008 [0x1111 : SB[3] = RL;;0x2222 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 2nd instance"):
        sb3["0x1111"] <= RL()
        RL ["0x2222"] <= sb6() | NRL()  # uses old RL

    #     009 [0x2222 : SB[6] = RL;;0x1111 : RL = SB[5];]
    with apl_commands("compatible sections: 3rd instance"):
        sb6["0x2222"] <= RL()
        RL ["0x1111"] <= sb5()

    #     010 [0x2222 : RL = SB[5] & NRL;]
    RL["0x2222"] <= sb5() & NRL()

    #     011 [0x2222 : SB[5] = RL;;0x4444 : RL = SB[5];]
    with apl_commands("compatible sections: 4th instance"):
        sb5["0x2222"] <= RL()
        RL ["0x4444"] <= sb5()

    #     012 [0x2222 : RL = SB[6] & SRL;]
    RL["0x2222"] <= sb6() & SRL()

    #     013 [0x2222 : SB[3] = RL;;0x4444 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 5th instance"):
        sb3["0x2222"] <= RL()
        RL ["0x4444"] <= sb6() | NRL()

    #     014 [0x4444 : SB[6] = RL;;0x2222 : RL = SB[5];]
    with apl_commands("compatible sections: 6th instance"):
        sb6["0x4444"] <= RL()
        RL ["0x2222"] <= sb5()

    #     015 [0x4444 : RL = SB[5] & NRL;]
    RL["0x4444"] <= sb5() & NRL()

    #     016 [0x4444 : SB[5] = RL;;0x8888 : RL = SB[5];]
    with apl_commands("compatible sections: 7th instance"):
        sb5["0x4444"] <= RL()
        RL ["0x8888"] <= sb5()

    #     017 [0x4444 : RL = SB[6] & SRL;]
    RL["0x4444"] <= sb6() & SRL()

    #     018 [0x4444 : SB[3] = RL;;0x8888 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 8th instance"):
        sb3["0x4444"] <= RL()
        RL ["0x8888"] <= sb6() | NRL()

    #     019 [0x8888 : SB[6] = RL;;0x4444 : RL = SB[5];]
    with apl_commands("compatible sections: 9th instance"):
        sb6["0x8888"] <= RL()
        RL ["0x4444"] <= sb5()

    #     020 [0x8888 : RL = SB[5] & NRL;]
    RL["0x8888"] <= sb5() & NRL()

    #    021 [0x8888 : SB[3,5] = RL;]
    eights = Belex.Mask("0x8888")
    eights[sb3, sb5] <= RL()

    #     022 [0x000f : RL = SB[5,7];]
    RL["0x000F"] <= sb5() & sb7()

    #     023 [0x000f : RL |= SB[6];]
    RL["0x000F"] |= sb6()

    #     024 [0x000f : SB[3,6] = RL;]
    x000f = Belex.Mask("0x000f")
    x000f[sb3, sb6] <= RL()

    #     025 [0x0008 : RL = SB[6];;0x0008 : GL = RL;]
    with apl_commands("broadcast after read: 1st instance"):
        RL["0x0008"] <= sb6()
        GL["0x0008"] <= RL()  # uses NEW RL

    #     026 [0xffff : SB[4] = GL;;0x0007 : RL = SB[6];]
    with apl_commands("read after write: 1st instance"):
        sb4[   ::   ] <= GL()
        RL ["0x0007"] <= sb6()

    #     027 [0x000e : SB[6] = NRL;;0x0001 : RL = SB[7];]
    with apl_commands("compatible sections: 10th instance"):
        sb6["0x000e"] <= NRL()
        RL ["0x0001"] <= sb7()

    #     028 [0x0001 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 2nd instance"):
        sb6["0x0001"] <= RL()
        RL [   ::   ] <= sb4()

    #     029 [0xffff : SB[7] = RL;]
    sb7[:] <= RL()

    #     030 [0x00f0 : RL = SB[5,7];]
    RL["0x00f0"] <= sb5() & sb7()

    #     031 [0x00f0 : RL |= SB[6];]
    RL["0x00f0"] |= sb6()

    #     032 [0x00f0 : SB[3,6] = RL;]
    x00f0 = Belex.Mask("0x00f0")
    x00f0[sb3, sb6] <= RL()

    #     033 [0x0080 : RL = SB[6];;0x0080 : GL = RL;]
    with apl_commands("broadcast after read: 2nd instance"):
        RL["0x0080"] <= sb6()
        GL["0x0080"] <= RL()

    #     034 [0xffff : SB[4] = GL;;0x0070 : RL = SB[6];]
    with apl_commands("read after write: 3rd instance"):
        sb4[   ::   ] <= GL()
        RL ["0x0070"] <= sb6()

    #     035 [0x00e0 : SB[6] = NRL;;0x0010 : RL = SB[7];]
    with apl_commands("compatible sections: 11th instance"):
        sb6["0x00e0"] <= NRL()
        RL ["0x0010"] <= sb7()

    #     036 [0x0010 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 4th instance"):
        sb6["0x0010"] <= RL()
        RL [   ::   ] <= sb4()

    #     037 [0xffff : SB[7] = RL;]
    sb7[::] = RL()

    #     038 [0x0f00 : RL = SB[5,7];]  # OPPORTUNITY?
    RL["0x0f00"] <= sb5() & sb7()

    #     039 [0x0f00 : RL |= SB[6];]
    RL["0x0f00"] |= sb6()

    #     040 [0x0f00 : SB[3,6] = RL;]
    x0f00 = Belex.Mask("0x0f00")
    x0f00[sb3, sb6] <= RL()

    #     041 [0x0800 : RL = SB[6];;0x0800 : GL = RL;]
    with apl_commands("broadcast after read: 3rd instance"):
        RL["0x0800"] <= sb6()
        GL["0x0800"] <= RL()

    #     042 [0xffff : SB[4] = GL;;0x0700 : RL = SB[6];]
    with apl_commands("read after write: 4th instance"):
        sb4[   ::   ] <= GL()
        RL ["0x0700"] <= sb6()

    #     043 [0x0e00 : SB[6] = NRL;;0x0100 : RL = SB[7];]
    with apl_commands("compatible sections: 12th instance"):
        sb6["0x0e00"] <= NRL()
        RL ["0x0100"] <= sb7()

    #     044 [0x0100 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x0100"] <= RL()
        RL [   ::   ] <= sb4()

    #     045 [0xffff : SB[7] = RL;]
    sb7[::] = RL()

    #     046 [0xf000 : RL = SB[5,7];]
    RL["0xf000"] <= sb5() & sb7()

    #     047 [0xf000 : RL |= SB[6];]
    RL["0xf000"] |= sb6()

    #     048 [0xf000 : SB[6] = RL;]
    sb6["0xf000"] <= RL()

    #     049 [0x8000 : RL = SB[6];;0x8000 : GL = RL;]
    with apl_commands("broadcast after read: 4th instance"):
        RL["0x8000"] <= sb6()
        GL["0x8000"] <= RL()

    #     050 [0xffff : SB[4] = GL;;0x7000 : RL = SB[6];]
    with apl_commands("read after write: 4th instance"):
        sb4[   ::   ] <= GL()
        RL ["0x7000"] <= sb6()

    #     051 [0xe000 : SB[6] = NRL;;0x1000 : RL = SB[7];]
    with apl_commands("compatible sections: 13th instance"):
        sb6["0xe000"] <= NRL()
        RL ["0x1000"] <= sb7()

    #     052 [0x1000 : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x1000"] <= RL()
        RL [   ::   ] <= sb1()

    #     053 [0xffff : RL ^= SB[6];]
    RL[::] ^= sb6()

    #     054 [0xffff : SB[3] = RL;;0xffff : RL = SB[2];]
    with apl_commands("read after write: 6th instance"):
        sb3[:] <= RL()
        RL [:] <= sb2()

    #     055 [0xffff : RL ^= SB[3];]
    RL[:] ^= sb3()

    #     056 [0xffff : SB[0,3] = RL;;0xffff : RL = SB[4];]
    # We know why this doesn't work: out = SB[0], sb3 = SB[11]

    # REMOVE DEAD WRITE
    # xffff[sb3] <= RL()

    xffff[out] <= RL()

    # REMOVE DEAD READ
    # RL[:] <= sb4()

    #     057 [0xffff : SB[7] = RL;]

    # REMOVE DEAD WRITE
    # sb7[:] <= RL()



@belex_apl
def laning_laboratory_002(Belex, out: VR, sb1: VR, sb2: VR):

    sb3 = RN_REG_T3
    sb4 = RN_REG_T4
    sb5 = RN_REG_T5
    sb6 = RN_REG_T6
    sb7 = RN_REG_T0

    # Trick to zero-out RL (2 cycles)
    #     001 [0xffff : RL = SB[1];]
    #     002 [0xffff : RL ^= SB[1];]

    # FIRST OPTIMIZATION AFTER BASELINE
    # RL[:] <= sb1()  # 001 from original.txt
    # RL[:] ^= sb1()
    RL[:] <= 0

    #     003 [0xffff : SB[7] = RL;;0xffff : RL = SB[1,2];]
    with apl_commands("read after write, 1st instance"):
        sb7[:] <= RL()
        RL [:] <= sb1() & sb2()

    #     004 [0xffff : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write, 2nd instance"):
        sb6[:] <= RL()
        RL [:] <= sb1()

    #     005 [0xffff : RL ^= SB[2];]
    RL[:] ^= sb2()

    #     006 [0xffff : SB[3,5] = RL;]
    xffff = Belex.Mask("0xffff")
    xffff[sb3, sb5] <= RL()

    #     007 [0x2222 : RL = SB[5];;0x1111 : RL = SB[6] & SRL;]
    with apl_commands("compatible sections: 1st instance"):
        RL["0x2222"] <= sb5()
        RL["0x1111"] <= sb6() & SRL()  # uses old RL

    #     008 [0x1111 : SB[3] = RL;;0x2222 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 2nd instance"):
        sb3["0x1111"] <= RL()
        RL ["0x2222"] <= sb6() | NRL()  # uses old RL

    #     009 [0x2222 : SB[6] = RL;;0x1111 : RL = SB[5];]
    with apl_commands("compatible sections: 3rd instance"):
        sb6["0x2222"] <= RL()
        RL ["0x1111"] <= sb5()

    #     010 [0x2222 : RL = SB[5] & NRL;]
    RL["0x2222"] <= sb5() & NRL()

    #     011 [0x2222 : SB[5] = RL;;0x4444 : RL = SB[5];]
    with apl_commands("compatible sections: 4th instance"):
        sb5["0x2222"] <= RL()
        RL ["0x4444"] <= sb5()

    #     012 [0x2222 : RL = SB[6] & SRL;]
    RL["0x2222"] <= sb6() & SRL()

    #     013 [0x2222 : SB[3] = RL;;0x4444 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 5th instance"):
        sb3["0x2222"] <= RL()
        RL ["0x4444"] <= sb6() | NRL()

    #     014 [0x4444 : SB[6] = RL;;0x2222 : RL = SB[5];]
    with apl_commands("compatible sections: 6th instance"):
        sb6["0x4444"] <= RL()
        RL ["0x2222"] <= sb5()

    #     015 [0x4444 : RL = SB[5] & NRL;]
    RL["0x4444"] <= sb5() & NRL()

    #     016 [0x4444 : SB[5] = RL;;0x8888 : RL = SB[5];]
    with apl_commands("compatible sections: 7th instance"):
        sb5["0x4444"] <= RL()
        RL ["0x8888"] <= sb5()

    #     017 [0x4444 : RL = SB[6] & SRL;]
    RL["0x4444"] <= sb6() & SRL()

    #     018 [0x4444 : SB[3] = RL;;0x8888 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 8th instance"):
        sb3["0x4444"] <= RL()
        RL ["0x8888"] <= sb6() | NRL()

    #     019 [0x8888 : SB[6] = RL;;0x4444 : RL = SB[5];]
    with apl_commands("compatible sections: 9th instance"):
        sb6["0x8888"] <= RL()
        RL ["0x4444"] <= sb5()

    #     020 [0x8888 : RL = SB[5] & NRL;]
    RL["0x8888"] <= sb5() & NRL()

    #    021 [0x8888 : SB[3,5] = RL;]
    eights = Belex.Mask("0x8888")
    eights[sb3, sb5] <= RL()

    #     022 [0x000f : RL = SB[5,7];]
    RL["0x000F"] <= sb5() & sb7()

    #     023 [0x000f : RL |= SB[6];]
    RL["0x000F"] |= sb6()

    #     024 [0x000f : SB[3,6] = RL;]
    x000f = Belex.Mask("0x000f")
    x000f[sb3, sb6] <= RL()

    #     025 [0x0008 : RL = SB[6];;0x0008 : GL = RL;]
    with apl_commands("broadcast after read: 1st instance"):
        RL["0x0008"] <= sb6()
        GL["0x0008"] <= RL()  # uses NEW RL

    #     026 [0xffff : SB[4] = GL;;0x0007 : RL = SB[6];]
    with apl_commands("read after write: 1st instance"):
        sb4[   ::   ] <= GL()
        RL ["0x0007"] <= sb6()

    #     027 [0x000e : SB[6] = NRL;;0x0001 : RL = SB[7];]
    with apl_commands("compatible sections: 10th instance"):
        sb6["0x000e"] <= NRL()
        RL ["0x0001"] <= sb7()

    #     028 [0x0001 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 2nd instance"):
        sb6["0x0001"] <= RL()
        RL [   ::   ] <= sb4()

    #     029 [0xffff : SB[7] = RL;]
    sb7[:] <= RL()

    #     030 [0x00f0 : RL = SB[5,7];]
    RL["0x00f0"] <= sb5() & sb7()

    #     031 [0x00f0 : RL |= SB[6];]
    RL["0x00f0"] |= sb6()

    #     032 [0x00f0 : SB[3,6] = RL;]
    x00f0 = Belex.Mask("0x00f0")
    x00f0[sb3, sb6] <= RL()

    #     033 [0x0080 : RL = SB[6];;0x0080 : GL = RL;]
    with apl_commands("broadcast after read: 2nd instance"):
        RL["0x0080"] <= sb6()
        GL["0x0080"] <= RL()

    #     034 [0xffff : SB[4] = GL;;0x0070 : RL = SB[6];]
    with apl_commands("read after write: 3rd instance"):
        sb4[   ::   ] <= GL()
        RL ["0x0070"] <= sb6()

    #     035 [0x00e0 : SB[6] = NRL;;0x0010 : RL = SB[7];]
    with apl_commands("compatible sections: 11th instance"):
        sb6["0x00e0"] <= NRL()
        RL ["0x0010"] <= sb7()

    #     036 [0x0010 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 4th instance"):
        sb6["0x0010"] <= RL()
        RL [   ::   ] <= sb4()

    #     037 [0xffff : SB[7] = RL;]
    sb7[::] = RL()

    #     038 [0x0f00 : RL = SB[5,7];]  # OPPORTUNITY?
    RL["0x0f00"] <= sb5() & sb7()

    #     039 [0x0f00 : RL |= SB[6];]
    RL["0x0f00"] |= sb6()

    #     040 [0x0f00 : SB[3,6] = RL;]
    x0f00 = Belex.Mask("0x0f00")
    x0f00[sb3, sb6] <= RL()

    #     041 [0x0800 : RL = SB[6];;0x0800 : GL = RL;]
    with apl_commands("broadcast after read: 3rd instance"):
        RL["0x0800"] <= sb6()
        GL["0x0800"] <= RL()

    #     042 [0xffff : SB[4] = GL;;0x0700 : RL = SB[6];]
    with apl_commands("read after write: 4th instance"):
        sb4[   ::   ] <= GL()
        RL ["0x0700"] <= sb6()

    #     043 [0x0e00 : SB[6] = NRL;;0x0100 : RL = SB[7];]
    with apl_commands("compatible sections: 12th instance"):
        sb6["0x0e00"] <= NRL()
        RL ["0x0100"] <= sb7()

    #     044 [0x0100 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x0100"] <= RL()
        RL [   ::   ] <= sb4()

    #     045 [0xffff : SB[7] = RL;]
    sb7[::] = RL()

    #     046 [0xf000 : RL = SB[5,7];]
    RL["0xf000"] <= sb5() & sb7()

    #     047 [0xf000 : RL |= SB[6];]
    RL["0xf000"] |= sb6()

    #     048 [0xf000 : SB[6] = RL;]
    sb6["0xf000"] <= RL()

    #     049 [0x8000 : RL = SB[6];;0x8000 : GL = RL;]
    with apl_commands("broadcast after read: 4th instance"):
        RL["0x8000"] <= sb6()
        GL["0x8000"] <= RL()

    #     050 [0xffff : SB[4] = GL;;0x7000 : RL = SB[6];]
    with apl_commands("read after write: 4th instance"):
        sb4[   ::   ] <= GL()
        RL ["0x7000"] <= sb6()

    #     051 [0xe000 : SB[6] = NRL;;0x1000 : RL = SB[7];]
    with apl_commands("compatible sections: 13th instance"):
        sb6["0xe000"] <= NRL()
        RL ["0x1000"] <= sb7()

    #     052 [0x1000 : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x1000"] <= RL()
        RL [   ::   ] <= sb1()

    #     053 [0xffff : RL ^= SB[6];]
    RL[::] ^= sb6()

    #     054 [0xffff : SB[3] = RL;;0xffff : RL = SB[2];]
    with apl_commands("read after write: 6th instance"):
        # REMOVE sb3[:] <= RL()
        RL [:] ^= sb2()

    #     055 [0xffff : RL ^= SB[3];]
    # REMOVE
    # RL[:] ^= sb3()

    #     056 [0xffff : SB[0,3] = RL;;0xffff : RL = SB[4];]
    # We know why this doesn't work: out = SB[0], sb3 = SB[11]

    # REMOVE DEAD WRITE
    # xffff[sb3] <= RL()

    xffff[out] <= RL()

    # REMOVE DEAD READ
    # RL[:] <= sb4()

    #     057 [0xffff : SB[7] = RL;]

    # REMOVE DEAD WRITE
    # sb7[:] <= RL()



@belex_apl
def laning_laboratory_003(Belex, out: VR, sb1: VR, sb2: VR):

    sb3 = RN_REG_T3
    sb4 = RN_REG_T4
    sb5 = RN_REG_T5
    sb6 = RN_REG_T6
    sb7 = RN_REG_T0

    # Trick to zero-out RL (2 cycles)
    #     001 [0xffff : RL = SB[1];]
    #     002 [0xffff : RL ^= SB[1];]

    # FIRST OPTIMIZATION AFTER BASELINE
    # RL[:] <= sb1()  # 001 from original.txt
    # RL[:] ^= sb1()
    RL[:] <= 0

    #     003 [0xffff : SB[7] = RL;;0xffff : RL = SB[1,2];]
    with apl_commands("read after write, 1st instance"):
        sb7[:] <= RL()
        RL [:] <= sb1() & sb2()

    #     004 [0xffff : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write, 2nd instance"):
        sb6[:] <= RL()
        RL [:] <= sb1()

    #     005 [0xffff : RL ^= SB[2];]
    RL[:] ^= sb2()

    #     006 [0xffff : SB[3,5] = RL;]
    xffff = Belex.Mask("0xffff")
    xffff[sb3, sb5] <= RL()

    #     007 [0x2222 : RL = SB[5];;0x1111 : RL = SB[6] & SRL;]
    with apl_commands("compatible sections: 1st instance"):
        RL["0x2222"] <= sb5()
        RL["0x1111"] <= sb6() & SRL()  # uses old RL

    #     008 [0x1111 : SB[3] = RL;;0x2222 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 2nd instance"):
        sb3["0x1111"] <= RL()
        RL ["0x2222"] <= sb6() | NRL()  # uses old RL

    #     009 [0x2222 : SB[6] = RL;;0x1111 : RL = SB[5];]
    with apl_commands("compatible sections: 3rd instance"):
        sb6["0x2222"] <= RL()
        RL ["0x1111"] <= sb5()

    #     010 [0x2222 : RL = SB[5] & NRL;]
    RL["0x2222"] <= sb5() & NRL()

    #     011 [0x2222 : SB[5] = RL;;0x4444 : RL = SB[5];]
    with apl_commands("compatible sections: 4th instance"):
        sb5["0x2222"] <= RL()
        RL ["0x4444"] <= sb5()

    #     012 [0x2222 : RL = SB[6] & SRL;]
    RL["0x2222"] <= sb6() & SRL()

    #     013 [0x2222 : SB[3] = RL;;0x4444 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 5th instance"):
        sb3["0x2222"] <= RL()
        RL ["0x4444"] <= sb6() | NRL()

    #     014 [0x4444 : SB[6] = RL;;0x2222 : RL = SB[5];]
    with apl_commands("compatible sections: 6th instance"):
        sb6["0x4444"] <= RL()
        RL ["0x2222"] <= sb5()

    #     015 [0x4444 : RL = SB[5] & NRL;]
    RL["0x4444"] <= sb5() & NRL()

    #     016 [0x4444 : SB[5] = RL;;0x8888 : RL = SB[5];]
    with apl_commands("compatible sections: 7th instance"):
        sb5["0x4444"] <= RL()
        RL ["0x8888"] <= sb5()

    #     017 [0x4444 : RL = SB[6] & SRL;]
    RL["0x4444"] <= sb6() & SRL()

    #     018 [0x4444 : SB[3] = RL;;0x8888 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 8th instance"):
        sb3["0x4444"] <= RL()
        RL ["0x8888"] <= sb6() | NRL()

    #     019 [0x8888 : SB[6] = RL;;0x4444 : RL = SB[5];]
    with apl_commands("compatible sections: 9th instance"):
        sb6["0x8888"] <= RL()
        RL ["0x4444"] <= sb5()

    #     020 [0x8888 : RL = SB[5] & NRL;]
    RL["0x8888"] <= sb5() & NRL()

    #    021 [0x8888 : SB[3,5] = RL;]
    eights = Belex.Mask("0x8888")
    eights[sb3, sb5] <= RL()

    #     022 [0x000f : RL = SB[5,7];]
    RL["0x000F"] <= sb5() & sb7()

    #     023 [0x000f : RL |= SB[6];]
    RL["0x000F"] |= sb6()

    #     024 [0x000f : SB[3,6] = RL;]
    x000f = Belex.Mask("0x000f")
    x000f[sb3, sb6] <= RL()

    #     025 [0x0008 : RL = SB[6];;0x0008 : GL = RL;]
    with apl_commands("broadcast after read: 1st instance"):
        RL["0x0008"] <= sb6()
        GL["0x0008"] <= RL()  # uses NEW RL

    #     026 [0xffff : SB[4] = GL;;0x0007 : RL = SB[6];]
    with apl_commands("read after write: 1st instance"):
        sb4[   ::   ] <= GL()
        RL ["0x0007"] <= sb6()

    #     027 [0x000e : SB[6] = NRL;;0x0001 : RL = SB[7];]
    with apl_commands("compatible sections: 10th instance"):
        sb6["0x000e"] <= NRL()
        RL ["0x0001"] <= sb7()

    #     028 [0x0001 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 2nd instance"):
        sb6["0x0001"] <= RL()
        RL [   ::   ] <= sb4()

    #     029 [0xffff : SB[7] = RL;]
    sb7[:] <= RL()

    #     030 [0x00f0 : RL = SB[5,7];]
    RL["0x00f0"] <= sb5() & sb7()

    #     031 [0x00f0 : RL |= SB[6];]
    RL["0x00f0"] |= sb6()

    #     032 [0x00f0 : SB[3,6] = RL;]
    x00f0 = Belex.Mask("0x00f0")
    x00f0[sb3, sb6] <= RL()

    #     033 [0x0080 : RL = SB[6];;0x0080 : GL = RL;]
    with apl_commands("broadcast after read: 2nd instance"):
        RL["0x0080"] <= sb6()
        GL["0x0080"] <= RL()

    #     034 [0xffff : SB[4] = GL;;0x0070 : RL = SB[6];]
    with apl_commands("read after write: 3rd instance"):
        sb4[   ::   ] <= GL()
        RL ["0x0070"] <= sb6()

    #     035 [0x00e0 : SB[6] = NRL;;0x0010 : RL = SB[7];]
    with apl_commands("compatible sections: 11th instance"):
        sb6["0x00e0"] <= NRL()
        RL ["0x0010"] <= sb7()

    #     036 [0x0010 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 4th instance"):
        sb6["0x0010"] <= RL()
        RL [   ::   ] <= sb4()

    #     037 [0xffff : SB[7] = RL;]
    sb7[::] = RL()

    #     038 [0x0f00 : RL = SB[5,7];]  # OPPORTUNITY?
    RL["0x0f00"] <= sb5() & sb7()

    #     039 [0x0f00 : RL |= SB[6];]
    RL["0x0f00"] |= sb6()

    #     040 [0x0f00 : SB[3,6] = RL;]
    x0f00 = Belex.Mask("0x0f00")
    x0f00[sb3, sb6] <= RL()

    #     041 [0x0800 : RL = SB[6];;0x0800 : GL = RL;]
    with apl_commands("broadcast after read: 3rd instance"):
        RL["0x0800"] <= sb6()
        GL["0x0800"] <= RL()

    #     042 [0xffff : SB[4] = GL;;0x0700 : RL = SB[6];]
    with apl_commands("read after write: 4th instance"):
        sb4[   ::   ] <= GL()
        RL ["0x0700"] <= sb6()

    #     043 [0x0e00 : SB[6] = NRL;;0x0100 : RL = SB[7];]
    with apl_commands("compatible sections: 12th instance"):
        sb6["0x0e00"] <= NRL()
        RL ["0x0100"] <= sb7()

    #     044 [0x0100 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x0100"] <= RL()
        RL [   ::   ] <= sb4()

    #     045 [0xffff : SB[7] = RL;]
    sb7[::] = RL()

    #     046 [0xf000 : RL = SB[5,7];]
    RL["0xf000"] <= sb5() & sb7()

    #     047 [0xf000 : RL |= SB[6];]
    RL["0xf000"] |= sb6()

    #     048 [0xf000 : SB[6] = RL;]
    sb6["0xf000"] <= RL()

    #     049 [0x8000 : RL = SB[6];;0x8000 : GL = RL;]
    with apl_commands("broadcast after read: 4th instance"):
        RL["0x8000"] <= sb6()
        GL["0x8000"] <= RL()

    #     050 [0xffff : SB[4] = GL;;0x7000 : RL = SB[6];]
    with apl_commands("read after write: 4th instance"):
        sb4[   ::   ] <= GL()
        RL ["0x7000"] <= sb6()

    #     051 [0xe000 : SB[6] = NRL;;0x1000 : RL = SB[7];]
    with apl_commands("compatible sections: 13th instance"):
        sb6["0xe000"] <= NRL()
        RL ["0x1000"] <= sb7()

    #     052 [0x1000 : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x1000"] <= RL()
        RL [   ::   ] <= sb1()

    #     053 [0xffff : RL ^= SB[6];]
    RL[::] ^= sb6()

    #     054 [0xffff : SB[3] = RL;;0xffff : RL = SB[2];]
    with apl_commands("read after write: 6th instance"):
        # 1. Remove unnecessary spill of RL and then use it immediately after
        # 2. Exploit commutativity of XOR
        # 3. Followed by dead-write elimination
        # REMOVE sb3[:] <= RL()  #
        # CHANGE RL [:] <= sb2()
        RL [:] ^= sb2()

    #     055 [0xffff : RL ^= SB[3];]
    # REMOVE
    # RL[:] ^= sb3()

    #     056 [0xffff : SB[0,3] = RL;;0xffff : RL = SB[4];]
    # We know why this doesn't work: out = SB[0], sb3 = SB[11]

    # REMOVE DEAD WRITE
    # xffff[sb3] <= RL()

    xffff[out] <= RL()

    # REMOVE DEAD READ
    # RL[:] <= sb4()

    #     057 [0xffff : SB[7] = RL;]

    # REMOVE DEAD WRITE
    # sb7[:] <= RL()



@belex_apl
def laning_laboratory_004(Belex, out: VR, sb1: VR, sb2: VR):

    sb3 = RN_REG_T3
    sb4 = RN_REG_T4
    sb5 = RN_REG_T5
    sb6 = RN_REG_T6
    sb7 = RN_REG_T0

    # Trick to zero-out RL (2 cycles)
    #     001 [0xffff : RL = SB[1];]
    #     002 [0xffff : RL ^= SB[1];]

    # FIRST OPTIMIZATION AFTER BASELINE
    # RL[:] <= sb1()  # 001 from original.txt
    # RL[:] ^= sb1()
    RL[:] <= 0

    #     003 [0xffff : SB[7] = RL;;0xffff : RL = SB[1,2];]
    with apl_commands("read after write, 1st instance"):
        sb7[:] <= RL()
        RL [:] <= sb1() & sb2()

    #     004 [0xffff : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write, 2nd instance"):
        sb6[:] <= RL()
        RL [:] <= sb1()

    #     005 [0xffff : RL ^= SB[2];]
    RL[:] ^= sb2()

    #     006 [0xffff : SB[3,5] = RL;]
    xffff = Belex.Mask("0xffff")
    xffff[sb3, sb5] <= RL()

    #     007 [0x2222 : RL = SB[5];;0x1111 : RL = SB[6] & SRL;]
    with apl_commands("compatible sections: 1st instance"):
        RL["0x2222"] <= sb5()
        RL["0x1111"] <= sb6() & SRL()  # uses old RL

    #     008 [0x1111 : SB[3] = RL;;0x2222 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 2nd instance"):
        sb3["0x1111"] <= RL()
        RL ["0x2222"] <= sb6() | NRL()  # uses old RL

    #     009 [0x2222 : SB[6] = RL;;0x1111 : RL = SB[5];]
    with apl_commands("compatible sections: 3rd instance"):
        sb6["0x2222"] <= RL()
        RL ["0x1111"] <= sb5()

    #     010 [0x2222 : RL = SB[5] & NRL;]
    RL["0x2222"] <= sb5() & NRL()

    #     011 [0x2222 : SB[5] = RL;;0x4444 : RL = SB[5];]
    with apl_commands("compatible sections: 4th instance"):
        sb5["0x2222"] <= RL()
        RL ["0x4444"] <= sb5()

    #     012 [0x2222 : RL = SB[6] & SRL;]
    RL["0x2222"] <= sb6() & SRL()

    #     013 [0x2222 : SB[3] = RL;;0x4444 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 5th instance"):
        sb3["0x2222"] <= RL()
        RL ["0x4444"] <= sb6() | NRL()

    #     014 [0x4444 : SB[6] = RL;;0x2222 : RL = SB[5];]
    with apl_commands("compatible sections: 6th instance"):
        sb6["0x4444"] <= RL()
        RL ["0x2222"] <= sb5()

    #     015 [0x4444 : RL = SB[5] & NRL;]
    RL["0x4444"] <= sb5() & NRL()

    #     016 [0x4444 : SB[5] = RL;;0x8888 : RL = SB[5];]
    with apl_commands("compatible sections: 7th instance"):
        sb5["0x4444"] <= RL()
        RL ["0x8888"] <= sb5()

    #     017 [0x4444 : RL = SB[6] & SRL;]
    RL["0x4444"] <= sb6() & SRL()

    #     018 [0x4444 : SB[3] = RL;;0x8888 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 8th instance"):
        sb3["0x4444"] <= RL()
        RL ["0x8888"] <= sb6() | NRL()

    #     019 [0x8888 : SB[6] = RL;;0x4444 : RL = SB[5];]
    with apl_commands("compatible sections: 9th instance"):
        sb6["0x8888"] <= RL()
        RL ["0x4444"] <= sb5()

    #     020 [0x8888 : RL = SB[5] & NRL;]
    RL["0x8888"] <= sb5() & NRL()

    #    021 [0x8888 : SB[3,5] = RL;]
    eights = Belex.Mask("0x8888")
    eights[sb3, sb5] <= RL()

    #     022 [0x000f : RL = SB[5,7];]
    RL["0x000F"] <= sb5() & sb7()

    #     023 [0x000f : RL |= SB[6];]
    RL["0x000F"] |= sb6()

    #     024 [0x000f : SB[3,6] = RL;]
    x000f = Belex.Mask("0x000f")
    x000f[sb3, sb6] <= RL()

    #     025 [0x0008 : RL = SB[6];;0x0008 : GL = RL;]
    with apl_commands("broadcast after read: 1st instance"):
        RL["0x0008"] <= sb6()
        GL["0x0008"] <= RL()  # uses NEW RL

    #     026 [0xffff : SB[4] = GL;;0x0007 : RL = SB[6];]
    with apl_commands("read after write: 1st instance"):
        sb4[   ::   ] <= GL()
        RL ["0x0007"] <= sb6()

    #     027 [0x000e : SB[6] = NRL;;0x0001 : RL = SB[7];]
    with apl_commands("compatible sections: 10th instance"):
        sb6["0x000e"] <= NRL()
        RL ["0x0001"] <= sb7()

    #     028 [0x0001 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 2nd instance"):
        sb6["0x0001"] <= RL()
        RL [   ::   ] <= sb4()

    #     029 [0xffff : SB[7] = RL;]
    sb7[:] <= RL()

    #     030 [0x00f0 : RL = SB[5,7];]
    RL["0x00f0"] <= sb5() & sb7()

    #     031 [0x00f0 : RL |= SB[6];]
    RL["0x00f0"] |= sb6()

    #     032 [0x00f0 : SB[3,6] = RL;]
    x00f0 = Belex.Mask("0x00f0")
    x00f0[sb3, sb6] <= RL()

    #     033 [0x0080 : RL = SB[6];;0x0080 : GL = RL;]
    with apl_commands("broadcast after read: 2nd instance"):
        RL["0x0080"] <= sb6()
        GL["0x0080"] <= RL()

    #     034 [0xffff : SB[4] = GL;;0x0070 : RL = SB[6];]
    with apl_commands("read after write: 3rd instance"):
        sb4[   ::   ] <= GL()
        RL ["0x0070"] <= sb6()

    #     035 [0x00e0 : SB[6] = NRL;;0x0010 : RL = SB[7];]
    with apl_commands("compatible sections: 11th instance"):
        sb6["0x00e0"] <= NRL()
        RL ["0x0010"] <= sb7()

    #     036 [0x0010 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 4th instance"):
        sb6["0x0010"] <= RL()
        RL [   ::   ] <= sb4()

    #     037 [0xffff : SB[7] = RL;]
    sb7[::] = RL()

    #     038 [0x0f00 : RL = SB[5,7];]  # OPPORTUNITY?
    RL["0x0f00"] <= sb5() & sb7()

    #     039 [0x0f00 : RL |= SB[6];]
    RL["0x0f00"] |= sb6()

    #     040 [0x0f00 : SB[3,6] = RL;]
    x0f00 = Belex.Mask("0x0f00")
    x0f00[sb3, sb6] <= RL()

    #     041 [0x0800 : RL = SB[6];;0x0800 : GL = RL;]
    with apl_commands("broadcast after read: 3rd instance"):
        RL["0x0800"] <= sb6()
        GL["0x0800"] <= RL()

    #     042 [0xffff : SB[4] = GL;;0x0700 : RL = SB[6];]
    with apl_commands("read after write: 4th instance"):
        sb4[   ::   ] <= GL()
        RL ["0x0700"] <= sb6()

    #     043 [0x0e00 : SB[6] = NRL;;0x0100 : RL = SB[7];]
    with apl_commands("compatible sections: 12th instance"):
        sb6["0x0e00"] <= NRL()
        RL ["0x0100"] <= sb7()

    #     044 [0x0100 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x0100"] <= RL()
        RL [   ::   ] <= sb4()

    #     045 [0xffff : SB[7] = RL;]
    sb7[::] = RL()

    #     046 [0xf000 : RL = SB[5,7];]
    RL["0xf000"] <= sb5() & sb7()

    #     047 [0xf000 : RL |= SB[6];]
    RL["0xf000"] |= sb6()

    #     048 [0xf000 : SB[6] = RL;]
    sb6["0xf000"] <= RL()

    #     049 [0x8000 : RL = SB[6];;0x8000 : GL = RL;]
    with apl_commands("broadcast after read: 4th instance"):
        RL["0x8000"] <= sb6()
        GL["0x8000"] <= RL()

    #     050 [0xffff : SB[4] = GL;;0x7000 : RL = SB[6];]
    with apl_commands("read after write: 4th instance"):
        # REMOVE DEAD WRITE
        # sb4[   ::   ] <= GL()
        RL ["0x7000"] <= sb6()

    #     051 [0xe000 : SB[6] = NRL;;0x1000 : RL = SB[7];]
    with apl_commands("compatible sections: 13th instance"):
        sb6["0xe000"] <= NRL()
        RL ["0x1000"] <= sb7()

    #     052 [0x1000 : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x1000"] <= RL()
        RL [   ::   ] <= sb1()

    #     053 [0xffff : RL ^= SB[6];]
    RL[::] ^= sb6()

    #     054 [0xffff : SB[3] = RL;;0xffff : RL = SB[2];]
    with apl_commands("read after write: 6th instance"):
        # 1. Remove unnecessary spill of RL and then use it immediately after
        # 2. Exploit commutativity of XOR
        # 3. Followed by dead-write elimination
        # REMOVE sb3[:] <= RL()  #
        # CHANGE RL [:] <= sb2()
        RL [:] ^= sb2()

    #     055 [0xffff : RL ^= SB[3];]
    # REMOVE
    # RL[:] ^= sb3()

    #     056 [0xffff : SB[0,3] = RL;;0xffff : RL = SB[4];]
    # We know why this doesn't work: out = SB[0], sb3 = SB[11]

    # REMOVE DEAD WRITE
    # xffff[sb3] <= RL()

    xffff[out] <= RL()

    # REMOVE DEAD READ
    # RL[:] <= sb4()

    #     057 [0xffff : SB[7] = RL;]

    # REMOVE DEAD WRITE
    # sb7[:] <= RL()



@belex_apl
def laning_laboratory_005(Belex, out: VR, sb1: VR, sb2: VR):

    sb3 = RN_REG_T3
    sb4 = RN_REG_T4
    sb5 = RN_REG_T5
    sb6 = RN_REG_T6
    sb7 = RN_REG_T0

    # Trick to zero-out RL (2 cycles)
    #     001 [0xffff : RL = SB[1];]
    #     002 [0xffff : RL ^= SB[1];]

    # FIRST OPTIMIZATION AFTER BASELINE
    # RL[:] <= sb1()  # 001 from original.txt
    # RL[:] ^= sb1()
    RL[:] <= 0

    #     003 [0xffff : SB[7] = RL;;0xffff : RL = SB[1,2];]
    with apl_commands("read after write, 1st instance"):
        sb7[:] <= RL()
        RL [:] <= sb1() & sb2()

    #     004 [0xffff : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write, 2nd instance"):
        sb6[:] <= RL()
        RL [:] <= sb1()

    #     005 [0xffff : RL ^= SB[2];]
    RL[:] ^= sb2()

    #     006 [0xffff : SB[3,5] = RL;]
    xffff = Belex.Mask("0xffff")
    xffff[sb3, sb5] <= RL()

    #     007 [0x2222 : RL = SB[5];;0x1111 : RL = SB[6] & SRL;]
    with apl_commands("compatible sections: 1st instance"):
        RL["0x2222"] <= sb5()
        RL["0x1111"] <= sb6() & SRL()  # uses old RL

    #     008 [0x1111 : SB[3] = RL;;0x2222 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 2nd instance"):
        sb3["0x1111"] <= RL()
        RL ["0x2222"] <= sb6() | NRL()  # uses old RL

    #     009 [0x2222 : SB[6] = RL;;0x1111 : RL = SB[5];]
    with apl_commands("compatible sections: 3rd instance"):
        sb6["0x2222"] <= RL()
        RL ["0x1111"] <= sb5()

    #     010 [0x2222 : RL = SB[5] & NRL;]
    RL["0x2222"] <= sb5() & NRL()

    #     011 [0x2222 : SB[5] = RL;;0x4444 : RL = SB[5];]
    with apl_commands("compatible sections: 4th instance"):
        sb5["0x2222"] <= RL()
        RL ["0x4444"] <= sb5()

    #     012 [0x2222 : RL = SB[6] & SRL;]
    RL["0x2222"] <= sb6() & SRL()

    #     013 [0x2222 : SB[3] = RL;;0x4444 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 5th instance"):
        sb3["0x2222"] <= RL()
        RL ["0x4444"] <= sb6() | NRL()

    #     014 [0x4444 : SB[6] = RL;;0x2222 : RL = SB[5];]
    with apl_commands("compatible sections: 6th instance"):
        sb6["0x4444"] <= RL()
        RL ["0x2222"] <= sb5()

    #     015 [0x4444 : RL = SB[5] & NRL;]
    RL["0x4444"] <= sb5() & NRL()

    #     016 [0x4444 : SB[5] = RL;;0x8888 : RL = SB[5];]
    with apl_commands("compatible sections: 7th instance"):
        sb5["0x4444"] <= RL()
        RL ["0x8888"] <= sb5()

    #     017 [0x4444 : RL = SB[6] & SRL;]
    RL["0x4444"] <= sb6() & SRL()

    #     018 [0x4444 : SB[3] = RL;;0x8888 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 8th instance"):
        sb3["0x4444"] <= RL()
        RL ["0x8888"] <= sb6() | NRL()

    #     019 [0x8888 : SB[6] = RL;;0x4444 : RL = SB[5];]
    with apl_commands("compatible sections: 9th instance"):
        sb6["0x8888"] <= RL()
        RL ["0x4444"] <= sb5()

    #     020 [0x8888 : RL = SB[5] & NRL;]
    RL["0x8888"] <= sb5() & NRL()

    #    021 [0x8888 : SB[3,5] = RL;]
    eights = Belex.Mask("0x8888")
    eights[sb3, sb5] <= RL()

    #     022 [0x000f : RL = SB[5,7];]
    RL["0x000F"] <= sb5() & sb7()

    #     023 [0x000f : RL |= SB[6];]
    RL["0x000F"] |= sb6()

    #     024 [0x000f : SB[3,6] = RL;]
    x000f = Belex.Mask("0x000f")
    with apl_commands():
        GL["0x0008"] <= RL()  # uses NEW RL  # MOVE UP AND MERGE LANES W PREV INSTR
        x000f[sb3, sb6] <= RL()

    # #     025 [0x0008 : RL = SB[6];;0x0008 : GL = RL;]
    # with apl_commands("broadcast after read: 1st instance"):
    #     # RL["0x0008"] <= sb6()
    #     GL["0x0008"] <= RL()  # uses NEW RL  # MOVE UP AND MERGE LANES W PREV INSTR

    #     026 [0xffff : SB[4] = GL;;0x0007 : RL = SB[6];]
    with apl_commands("read after write: 1st instance"):
        sb4[   ::   ] <= GL()
        # RL ["0x0007"] <= sb6()

    #     027 [0x000e : SB[6] = NRL;;0x0001 : RL = SB[7];]
    with apl_commands("compatible sections: 10th instance"):
        sb6["0x000e"] <= NRL()
        RL ["0x0001"] <= sb7()

    #     028 [0x0001 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 2nd instance"):
        sb6["0x0001"] <= RL()
        RL [   ::   ] <= sb4()

    #     029 [0xffff : SB[7] = RL;]
    sb7[:] <= RL()

    #     030 [0x00f0 : RL = SB[5,7];]
    RL["0x00f0"] <= sb5() & sb7()

    #     031 [0x00f0 : RL |= SB[6];]
    RL["0x00f0"] |= sb6()

    #     032 [0x00f0 : SB[3,6] = RL;]
    x00f0 = Belex.Mask("0x00f0")
    with apl_commands():
        GL["0x0080"] <= RL()  # MOVE UP AND MERGE LANE W PREV INSTR
        x00f0[sb3, sb6] <= RL()

    # #     033 [0x0080 : RL = SB[6];;0x0080 : GL = RL;]
    # with apl_commands("broadcast after read: 2nd instance"):
    #     # RL["0x0080"] <= sb6()
    #     GL["0x0080"] <= RL()  # MOVE UP AND MERGE LANE W PREV INSTR

    #     034 [0xffff : SB[4] = GL;;0x0070 : RL = SB[6];]
    with apl_commands("read after write: 3rd instance"):
        sb4[   ::   ] <= GL()
        # RL ["0x0070"] <= sb6()

    #     035 [0x00e0 : SB[6] = NRL;;0x0010 : RL = SB[7];]
    with apl_commands("compatible sections: 11th instance"):
        sb6["0x00e0"] <= NRL()
        RL ["0x0010"] <= sb7()

    #     036 [0x0010 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 4th instance"):
        sb6["0x0010"] <= RL()
        RL [   ::   ] <= sb4()

    #     037 [0xffff : SB[7] = RL;]
    sb7[::] = RL()

    #     038 [0x0f00 : RL = SB[5,7];]  # OPPORTUNITY?
    RL["0x0f00"] <= sb5() & sb7()

    #     039 [0x0f00 : RL |= SB[6];]
    RL["0x0f00"] |= sb6()

    #     040 [0x0f00 : SB[3,6] = RL;]
    x0f00 = Belex.Mask("0x0f00")
    with apl_commands():
        GL["0x0800"] <= RL()
        x0f00[sb3, sb6] <= RL()

    # #     041 [0x0800 : RL = SB[6];;0x0800 : GL = RL;]
    # with apl_commands("broadcast after read: 3rd instance"):
    #     # RL["0x0800"] <= sb6()
    #     GL["0x0800"] <= RL()  # MOVE UP ONE INSTR AND MERGE INTO LANE

    #     042 [0xffff : SB[4] = GL;;0x0700 : RL = SB[6];]
    with apl_commands("read after write: 4th instance"):
        sb4[   ::   ] <= GL()
        # RL ["0x0700"] <= sb6()

    #     043 [0x0e00 : SB[6] = NRL;;0x0100 : RL = SB[7];]
    with apl_commands("compatible sections: 12th instance"):
        sb6["0x0e00"] <= NRL()
        RL ["0x0100"] <= sb7()

    #     044 [0x0100 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x0100"] <= RL()
        RL [   ::   ] <= sb4()

    #     045 [0xffff : SB[7] = RL;]
    sb7[::] = RL()

    #     046 [0xf000 : RL = SB[5,7];]
    RL["0xf000"] <= sb5() & sb7()

    #     047 [0xf000 : RL |= SB[6];]
    RL["0xf000"] |= sb6()

    #     048 [0xf000 : SB[6] = RL;]
    sb6["0xf000"] <= RL()

    # #     049 [0x8000 : RL = SB[6];;0x8000 : GL = RL;]
    # with apl_commands("broadcast after read: 4th instance"):
    #     # MERGE MASKS of old RL["0x8000"] with RL["0x7000"] from OLD 50
    #     RL["0xF000"] <= sb6()
    #     # REMOVE GL; it's dead
    #     # GL["0x8000"] <= RL()

    # #     050 [0xffff : SB[4] = GL;;0x7000 : RL = SB[6];]
    # with apl_commands("read after write: 4th instance"):
    #     # REMOVE DEAD WRITE
    #     # sb4[   ::   ] <= GL()
    #     RL ["0x7000"] <= sb6()

    #     051 [0xe000 : SB[6] = NRL;;0x1000 : RL = SB[7];]
    with apl_commands("compatible sections: 13th instance"):
        sb6["0xe000"] <= NRL()
        RL ["0x1000"] <= sb7()

    #     052 [0x1000 : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x1000"] <= RL()
        RL [   ::   ] <= sb1()

    #     053 [0xffff : RL ^= SB[6];]
    RL[::] ^= sb6()

    #     054 [0xffff : SB[3] = RL;;0xffff : RL = SB[2];]
    with apl_commands("read after write: 6th instance"):
        # 1. REMOVE UNNECESSARY SPILL OF RL and then use it immediately after
        # 2. EXPLOIT COMMUTATIVITY OF XOR
        # 3. ELIMINATE DEAD WRITE
        # REMOVE sb3[:] <= RL()  #
        # CHANGE RL [:] <= sb2()
        RL [:] ^= sb2()

    #     055 [0xffff : RL ^= SB[3];]
    # REMOVE
    # RL[:] ^= sb3()

    #     056 [0xffff : SB[0,3] = RL;;0xffff : RL = SB[4];]
    # We know why this doesn't work: out = SB[0], sb3 = SB[11]

    # REMOVE DEAD WRITE
    # xffff[sb3] <= RL()

    xffff[out] <= RL()

    # REMOVE DEAD READ
    # RL[:] <= sb4()

    #     057 [0xffff : SB[7] = RL;]

    # REMOVE DEAD WRITE
    # sb7[:] <= RL()



@belex_apl
def laning_laboratory_006(Belex, out: VR, sb1: VR, sb2: VR):

    sb3 = RN_REG_T3
    # sb4 = RN_REG_T4
    sb5 = RN_REG_T5
    sb6 = RN_REG_T6
    sb7 = RN_REG_T0

    # Trick to zero-out RL (2 cycles)
    #     001 [0xffff : RL = SB[1];]
    #     002 [0xffff : RL ^= SB[1];]

    # FIRST OPTIMIZATION AFTER BASELINE
    # RL[:] <= sb1()  # 001 from original.txt
    # RL[:] ^= sb1()
    RL[:] <= 0

    #     003 [0xffff : SB[7] = RL;;0xffff : RL = SB[1,2];]
    with apl_commands("read after write, 1st instance"):
        sb7[:] <= RL()
        RL [:] <= sb1() & sb2()

    #     004 [0xffff : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write, 2nd instance"):
        sb6[:] <= RL()
        RL [:] <= sb1()

    #     005 [0xffff : RL ^= SB[2];]
    RL[:] ^= sb2()

    #     006 [0xffff : SB[3,5] = RL;]
    xffff = Belex.Mask("0xffff")
    xffff[sb3, sb5] <= RL()

    #     007 [0x2222 : RL = SB[5];;0x1111 : RL = SB[6] & SRL;]
    with apl_commands("compatible sections: 1st instance"):
        RL["0x2222"] <= sb5()
        RL["0x1111"] <= sb6() & SRL()  # uses old RL

    #     008 [0x1111 : SB[3] = RL;;0x2222 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 2nd instance"):
        sb3["0x1111"] <= RL()
        RL ["0x2222"] <= sb6() | NRL()  # uses old RL

    #     009 [0x2222 : SB[6] = RL;;0x1111 : RL = SB[5];]
    with apl_commands("compatible sections: 3rd instance"):
        sb6["0x2222"] <= RL()
        RL ["0x1111"] <= sb5()

    #     010 [0x2222 : RL = SB[5] & NRL;]
    RL["0x2222"] <= sb5() & NRL()

    #     011 [0x2222 : SB[5] = RL;;0x4444 : RL = SB[5];]
    with apl_commands("compatible sections: 4th instance"):
        sb5["0x2222"] <= RL()
        RL ["0x4444"] <= sb5()

    #     012 [0x2222 : RL = SB[6] & SRL;]
    RL["0x2222"] <= sb6() & SRL()

    #     013 [0x2222 : SB[3] = RL;;0x4444 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 5th instance"):
        sb3["0x2222"] <= RL()
        RL ["0x4444"] <= sb6() | NRL()

    #     014 [0x4444 : SB[6] = RL;;0x2222 : RL = SB[5];]
    with apl_commands("compatible sections: 6th instance"):
        sb6["0x4444"] <= RL()
        RL ["0x2222"] <= sb5()

    #     015 [0x4444 : RL = SB[5] & NRL;]
    RL["0x4444"] <= sb5() & NRL()

    #     016 [0x4444 : SB[5] = RL;;0x8888 : RL = SB[5];]
    with apl_commands("compatible sections: 7th instance"):
        sb5["0x4444"] <= RL()
        RL ["0x8888"] <= sb5()

    #     017 [0x4444 : RL = SB[6] & SRL;]
    RL["0x4444"] <= sb6() & SRL()

    #     018 [0x4444 : SB[3] = RL;;0x8888 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 8th instance"):
        sb3["0x4444"] <= RL()
        RL ["0x8888"] <= sb6() | NRL()

    #     019 [0x8888 : SB[6] = RL;;0x4444 : RL = SB[5];]
    with apl_commands("compatible sections: 9th instance"):
        sb6["0x8888"] <= RL()
        RL ["0x4444"] <= sb5()

    #     020 [0x8888 : RL = SB[5] & NRL;]
    RL["0x8888"] <= sb5() & NRL()

    #    021 [0x8888 : SB[3,5] = RL;]
    eights = Belex.Mask("0x8888")
    eights[sb3, sb5] <= RL()

    #     022 [0x000f : RL = SB[5,7];]
    RL["0x000F"] <= sb5() & sb7()

    #     023 [0x000f : RL |= SB[6];]
    RL["0x000F"] |= sb6()

    #     024 [0x000f : SB[3,6] = RL;]
    x000f = Belex.Mask("0x000f")
    with apl_commands():
        GL["0x0008"] <= RL()  # uses NEW RL  # MOVE UP AND MERGE LANES W PREV INSTR
        x000f[sb3, sb6] <= RL()

    # #     025 [0x0008 : RL = SB[6];;0x0008 : GL = RL;]
    # with apl_commands("broadcast after read: 1st instance"):
    #     # RL["0x0008"] <= sb6()
    #     GL["0x0008"] <= RL()  # uses NEW RL  # MOVE UP AND MERGE LANES W PREV INSTR

    # #     026 [0xffff : SB[4] = GL;;0x0007 : RL = SB[6];]
    # with apl_commands("read after write: 1st instance"):
    #     sb4[   ::   ] <= GL()
    #     # RL ["0x0007"] <= sb6()

    #     027 [0x000e : SB[6] = NRL;;0x0001 : RL = SB[7];]
    with apl_commands("compatible sections: 10th instance"):
        sb6["0x000e"] <= NRL()
        RL ["0x0001"] <= sb7()

    #     028 [0x0001 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 2nd instance"):
        sb6["0x0001"] <= RL()
        # RL [   ::   ] <= sb4()

    #     029 [0xffff : SB[7] = RL;]
    # sb7[:] <= RL()
    sb7[:] <= GL()  # TODO: DYLON; SHOULD BE ABLE TO REMOVE THIS

    #     030 [0x00f0 : RL = SB[5,7];]
    RL["0x00f0"] <= sb5() & GL()  # sb7()

    #     031 [0x00f0 : RL |= SB[6];]
    RL["0x00f0"] |= sb6()

    #     032 [0x00f0 : SB[3,6] = RL;]
    x00f0 = Belex.Mask("0x00f0")
    with apl_commands():
        GL["0x0080"] <= RL()  # MOVE UP AND MERGE LANE W PREV INSTR
        x00f0[sb3, sb6] <= RL()

    # #     033 [0x0080 : RL = SB[6];;0x0080 : GL = RL;]
    # with apl_commands("broadcast after read: 2nd instance"):
    #     # RL["0x0080"] <= sb6()
    #     GL["0x0080"] <= RL()  # MOVE UP AND MERGE LANE W PREV INSTR

    # #     034 [0xffff : SB[4] = GL;;0x0070 : RL = SB[6];]
    # with apl_commands("read after write: 3rd instance"):
    #     sb4[   ::   ] <= GL()
    #     # RL ["0x0070"] <= sb6()

    #     035 [0x00e0 : SB[6] = NRL;;0x0010 : RL = SB[7];]
    with apl_commands("compatible sections: 11th instance"):
        sb6["0x00e0"] <= NRL()
        RL ["0x0010"] <= sb7()

    #     036 [0x0010 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 4th instance"):
        sb6["0x0010"] <= RL()
        # RL [   ::   ] <= sb4()

    #     037 [0xffff : SB[7] = RL;]
    sb7[:] = GL()  # RL()

    #     038 [0x0f00 : RL = SB[5,7];]  # OPPORTUNITY?
    RL["0x0f00"] <= sb5() & GL()  # sb7()

    #     039 [0x0f00 : RL |= SB[6];]
    RL["0x0f00"] |= sb6()

    #     040 [0x0f00 : SB[3,6] = RL;]
    x0f00 = Belex.Mask("0x0f00")
    with apl_commands():
        GL["0x0800"] <= RL()
        x0f00[sb3, sb6] <= RL()

    # #     041 [0x0800 : RL = SB[6];;0x0800 : GL = RL;]
    # with apl_commands("broadcast after read: 3rd instance"):
    #     # RL["0x0800"] <= sb6()
    #     GL["0x0800"] <= RL()  # MOVE UP ONE INSTR AND MERGE INTO LANE

    # #     042 [0xffff : SB[4] = GL;;0x0700 : RL = SB[6];]
    # with apl_commands("read after write: 4th instance"):
    #     sb4[   ::   ] <= GL()
    #     # RL ["0x0700"] <= sb6()

    #     043 [0x0e00 : SB[6] = NRL;;0x0100 : RL = SB[7];]
    with apl_commands("compatible sections: 12th instance"):
        sb6["0x0e00"] <= NRL()
        RL ["0x0100"] <= sb7()

    #     044 [0x0100 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x0100"] <= RL()
        # RL [   ::   ] <= sb4()

    #     045 [0xffff : SB[7] = RL;]
    sb7[:] = GL()  # RL()

    #     046 [0xf000 : RL = SB[5,7];]
    RL["0xf000"] <= sb5() & GL()  # sb7()

    #     047 [0xf000 : RL |= SB[6];]
    RL["0xf000"] |= sb6()

    #     048 [0xf000 : SB[6] = RL;]
    sb6["0xf000"] <= RL()

    # #     049 [0x8000 : RL = SB[6];;0x8000 : GL = RL;]
    # with apl_commands("broadcast after read: 4th instance"):
    #     # MERGE MASKS of old RL["0x8000"] with RL["0x7000"] from OLD 50
    #     RL["0xF000"] <= sb6()
    #     # REMOVE GL; it's dead
    #     # GL["0x8000"] <= RL()

    # #     050 [0xffff : SB[4] = GL;;0x7000 : RL = SB[6];]
    # with apl_commands("read after write: 4th instance"):
    #     # REMOVE DEAD WRITE
    #     # sb4[   ::   ] <= GL()
    #     RL ["0x7000"] <= sb6()

    #     051 [0xe000 : SB[6] = NRL;;0x1000 : RL = SB[7];]
    with apl_commands("compatible sections: 13th instance"):
        sb6["0xe000"] <= NRL()
        RL ["0x1000"] <= sb7()

    #     052 [0x1000 : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x1000"] <= RL()
        RL [   ::   ] <= sb1()

    #     053 [0xffff : RL ^= SB[6];]
    RL[::] ^= sb6()

    #     054 [0xffff : SB[3] = RL;;0xffff : RL = SB[2];]
    with apl_commands("read after write: 6th instance"):
        # 1. REMOVE UNNECESSARY SPILL OF RL and then use it immediately after
        # 2. EXPLOIT COMMUTATIVITY OF XOR
        # 3. ELIMINATE DEAD WRITE
        # REMOVE sb3[:] <= RL()  #
        # CHANGE RL [:] <= sb2()
        RL [:] ^= sb2()

    #     055 [0xffff : RL ^= SB[3];]
    # REMOVE
    # RL[:] ^= sb3()

    #     056 [0xffff : SB[0,3] = RL;;0xffff : RL = SB[4];]
    # We know why this doesn't work: out = SB[0], sb3 = SB[11]

    # REMOVE DEAD WRITE
    # xffff[sb3] <= RL()

    xffff[out] <= RL()

    # REMOVE DEAD READ
    # RL[:] <= sb4()

    #     057 [0xffff : SB[7] = RL;]

    # REMOVE DEAD WRITE
    # sb7[:] <= RL()



@belex_apl
def laning_laboratory_007(Belex, out: VR, sb1: VR, sb2: VR):

    sb3 = RN_REG_T3
    # sb4 = RN_REG_T4
    sb5 = RN_REG_T5
    sb6 = RN_REG_T6
    sb7 = RN_REG_T0

    # Trick to zero-out RL (2 cycles)
    #     001 [0xffff : RL = SB[1];]
    #     002 [0xffff : RL ^= SB[1];]

    # FIRST OPTIMIZATION AFTER BASELINE
    # RL[:] <= sb1()  # 001 from original.txt
    # RL[:] ^= sb1()
    RL[:] <= 0

    #     003 [0xffff : SB[7] = RL;;0xffff : RL = SB[1,2];]
    with apl_commands("read after write, 1st instance"):
        sb7[:] <= RL()
        RL [:] <= sb1() & sb2()

    #     004 [0xffff : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write, 2nd instance"):
        sb6[:] <= RL()
        RL [:] <= sb1()

    #     005 [0xffff : RL ^= SB[2];]
    RL[:] ^= sb2()

    #     006 [0xffff : SB[3,5] = RL;]
    xffff = Belex.Mask("0xffff")
    xffff[sb3, sb5] <= RL()

    #     007 [0x2222 : RL = SB[5];;0x1111 : RL = SB[6] & SRL;]
    with apl_commands("compatible sections: 1st instance"):
        RL["0x2222"] <= sb5()
        RL["0x1111"] <= sb6() & SRL()  # uses old RL

    #     008 [0x1111 : SB[3] = RL;;0x2222 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 2nd instance"):
        sb3["0x1111"] <= RL()
        RL ["0x2222"] <= sb6() | NRL()  # uses old RL

    #     009 [0x2222 : SB[6] = RL;;0x1111 : RL = SB[5];]
    with apl_commands("compatible sections: 3rd instance"):
        sb6["0x2222"] <= RL()
        RL ["0x1111"] <= sb5()

    #     010 [0x2222 : RL = SB[5] & NRL;]
    RL["0x2222"] <= sb5() & NRL()

    #     011 [0x2222 : SB[5] = RL;;0x4444 : RL = SB[5];]
    with apl_commands("compatible sections: 4th instance"):
        sb5["0x2222"] <= RL()
        RL ["0x4444"] <= sb5()

    #     012 [0x2222 : RL = SB[6] & SRL;]
    RL["0x2222"] <= sb6() & SRL()

    #     013 [0x2222 : SB[3] = RL;;0x4444 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 5th instance"):
        sb3["0x2222"] <= RL()
        RL ["0x4444"] <= sb6() | NRL()

    #     014 [0x4444 : SB[6] = RL;;0x2222 : RL = SB[5];]
    with apl_commands("compatible sections: 6th instance"):
        sb6["0x4444"] <= RL()
        RL ["0x2222"] <= sb5()

    #     015 [0x4444 : RL = SB[5] & NRL;]
    RL["0x4444"] <= sb5() & NRL()

    #     016 [0x4444 : SB[5] = RL;;0x8888 : RL = SB[5];]
    with apl_commands("compatible sections: 7th instance"):
        sb5["0x4444"] <= RL()
        RL ["0x8888"] <= sb5()

    #     017 [0x4444 : RL = SB[6] & SRL;]
    RL["0x4444"] <= sb6() & SRL()

    #     018 [0x4444 : SB[3] = RL;;0x8888 : RL = SB[6] | NRL;]
    with apl_commands("compatible sections: 8th instance"):
        sb3["0x4444"] <= RL()
        RL ["0x8888"] <= sb6() | NRL()

    #     019 [0x8888 : SB[6] = RL;;0x4444 : RL = SB[5];]
    with apl_commands("compatible sections: 9th instance"):
        sb6["0x8888"] <= RL()
        RL ["0x4444"] <= sb5()

    #     020 [0x8888 : RL = SB[5] & NRL;]
    RL["0x8888"] <= sb5() & NRL()

    #    021 [0x8888 : SB[3,5] = RL;]
    eights = Belex.Mask("0x8888")
    eights[sb3, sb5] <= RL()

    #     022 [0x000f : RL = SB[5,7];]
    RL["0x000F"] <= sb5() & sb7()

    #     023 [0x000f : RL |= SB[6];]
    RL["0x000F"] |= sb6()

    #     024 [0x000f : SB[3,6] = RL;]
    x000f = Belex.Mask("0x000f")
    with apl_commands():
        GL["0x0008"] <= RL()  # uses NEW RL  # MOVE UP AND MERGE LANES W PREV INSTR
        x000f[sb3, sb6] <= RL()

    # #     025 [0x0008 : RL = SB[6];;0x0008 : GL = RL;]
    # with apl_commands("broadcast after read: 1st instance"):
    #     # RL["0x0008"] <= sb6()
    #     GL["0x0008"] <= RL()  # uses NEW RL  # MOVE UP AND MERGE LANES W PREV INSTR

    # #     026 [0xffff : SB[4] = GL;;0x0007 : RL = SB[6];]
    # with apl_commands("read after write: 1st instance"):
    #     sb4[   ::   ] <= GL()
    #     # RL ["0x0007"] <= sb6()

    #     027 [0x000e : SB[6] = NRL;;0x0001 : RL = SB[7];]
    with apl_commands("compatible sections: 10th instance"):
        sb6["0x000e"] <= NRL()
        RL ["0x0001"] <= sb7()

    #     028 [0x0001 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 2nd instance"):
        sb6["0x0001"] <= RL()
        # RL [   ::   ] <= sb4()

    #     029 [0xffff : SB[7] = RL;]
    # sb7[:] <= RL()
    # sb7[:] <= GL()

    #     030 [0x00f0 : RL = SB[5,7];]
    with apl_commands():
        RL["0x00f0"] <= sb5() & GL()  # sb7()
        sb7[:] <= GL()

    #     031 [0x00f0 : RL |= SB[6];]
    RL["0x00f0"] |= sb6()

    #     032 [0x00f0 : SB[3,6] = RL;]
    x00f0 = Belex.Mask("0x00f0")
    with apl_commands():
        GL["0x0080"] <= RL()  # MOVE UP AND MERGE LANE W PREV INSTR
        x00f0[sb3, sb6] <= RL()

    # #     033 [0x0080 : RL = SB[6];;0x0080 : GL = RL;]
    # with apl_commands("broadcast after read: 2nd instance"):
    #     # RL["0x0080"] <= sb6()
    #     GL["0x0080"] <= RL()  # MOVE UP AND MERGE LANE W PREV INSTR

    # #     034 [0xffff : SB[4] = GL;;0x0070 : RL = SB[6];]
    # with apl_commands("read after write: 3rd instance"):
    #     sb4[   ::   ] <= GL()
    #     # RL ["0x0070"] <= sb6()

    #     035 [0x00e0 : SB[6] = NRL;;0x0010 : RL = SB[7];]
    with apl_commands("compatible sections: 11th instance"):
        sb6["0x00e0"] <= NRL()
        RL ["0x0010"] <= sb7()

    #     036 [0x0010 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 4th instance"):
        sb6["0x0010"] <= RL()
        # RL [   ::   ] <= sb4()

    #     037 [0xffff : SB[7] = RL;]

    #     038 [0x0f00 : RL = SB[5,7];]  # OPPORTUNITY?
    with apl_commands():
        RL["0x0f00"] <= sb5() & GL()  # sb7()
        sb7[:] = GL()  # RL()

    #     039 [0x0f00 : RL |= SB[6];]
    RL["0x0f00"] |= sb6()

    #     040 [0x0f00 : SB[3,6] = RL;]
    x0f00 = Belex.Mask("0x0f00")
    with apl_commands():
        GL["0x0800"] <= RL()
        x0f00[sb3, sb6] <= RL()

    # #     041 [0x0800 : RL = SB[6];;0x0800 : GL = RL;]
    # with apl_commands("broadcast after read: 3rd instance"):
    #     # RL["0x0800"] <= sb6()
    #     GL["0x0800"] <= RL()  # MOVE UP ONE INSTR AND MERGE INTO LANE

    # #     042 [0xffff : SB[4] = GL;;0x0700 : RL = SB[6];]
    # with apl_commands("read after write: 4th instance"):
    #     sb4[   ::   ] <= GL()
    #     # RL ["0x0700"] <= sb6()

    #     043 [0x0e00 : SB[6] = NRL;;0x0100 : RL = SB[7];]
    with apl_commands("compatible sections: 12th instance"):
        sb6["0x0e00"] <= NRL()
        RL ["0x0100"] <= sb7()

    #     044 [0x0100 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x0100"] <= RL()
        # RL [   ::   ] <= sb4()

    #     045 [0xffff : SB[7] = RL;]

    #     046 [0xf000 : RL = SB[5,7];]
    # UNREALISTIC TO REPLACE THE ALIAS OF GL FOR SB7
    with apl_commands():
        RL["0xf000"] <= sb5() & GL()  # sb7()
        # sb7[:] = GL()  # RL()

    #     047 [0xf000 : RL |= SB[6];]
    RL["0xf000"] |= sb6()

    #     048 [0xf000 : SB[6] = RL;]
    sb6["0xf000"] <= RL()

    # #     049 [0x8000 : RL = SB[6];;0x8000 : GL = RL;]
    # with apl_commands("broadcast after read: 4th instance"):
    #     # MERGE MASKS of old RL["0x8000"] with RL["0x7000"] from OLD 50
    #     RL["0xF000"] <= sb6()
    #     # REMOVE GL; it's dead
    #     # GL["0x8000"] <= RL()

    # #     050 [0xffff : SB[4] = GL;;0x7000 : RL = SB[6];]
    # with apl_commands("read after write: 4th instance"):
    #     # REMOVE DEAD WRITE
    #     # sb4[   ::   ] <= GL()
    #     RL ["0x7000"] <= sb6()

    #     051 [0xe000 : SB[6] = NRL;;0x1000 : RL = SB[7];]
    with apl_commands("compatible sections: 13th instance"):
        sb6["0xe000"] <= NRL()
        # RL ["0x1000"] <= sb7()

    #     052 [0x1000 : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x1000"] <= GL()  # RL()
        RL [   ::   ] <= sb1()

    #     053 [0xffff : RL ^= SB[6];]
    RL[::] ^= sb6()

    #     054 [0xffff : SB[3] = RL;;0xffff : RL = SB[2];]
    with apl_commands("read after write: 6th instance"):
        # 1. REMOVE UNNECESSARY SPILL OF RL and then use it immediately after
        # 2. EXPLOIT COMMUTATIVITY OF XOR
        # 3. ELIMINATE DEAD WRITE
        # REMOVE sb3[:] <= RL()  #
        # CHANGE RL [:] <= sb2()
        RL [:] ^= sb2()

    #     055 [0xffff : RL ^= SB[3];]
    # REMOVE
    # RL[:] ^= sb3()

    #     056 [0xffff : SB[0,3] = RL;;0xffff : RL = SB[4];]
    # We know why this doesn't work: out = SB[0], sb3 = SB[11]

    # REMOVE DEAD WRITE
    # xffff[sb3] <= RL()

    xffff[out] <= RL()

    # REMOVE DEAD READ
    # RL[:] <= sb4()

    #     057 [0xffff : SB[7] = RL;]

    # REMOVE DEAD WRITE
    # sb7[:] <= RL()



@belex_apl
def laning_laboratory_008(Belex, out: VR, sb1: VR, sb2: VR):

    # sb3 = RN_REG_T3  # discovered dead in this pass           #                     KILLS BY INSTRUCTION NUMBER
    # sb4 = RN_REG_T4  # discovered dead in earlier pass        # KILLS have arrows pointing IN to a node; GENS have arrows pointing out.
                                                                # Lack of a section mask means "all sections."
                                                                # In a KILL notation, the node indicates the value deposited.
                                                                # R* means that the READ (loading RL) occurs after a WRITE (loading SB).
                                                                # R* means that BROADCAST (loading GL) occurs after a READ (loading RL).
                                                                # Nodes on the same horizontal line are effected in parallel lanes.
                                                                # To figure out liveness of sections, do mask math in your head; look
                                                                #   upward in a column to find the bottom-most KILL of certain sections.
                                                                # The graph always contains dependencies from the original code. When I
                                                                #   make changes revealed by analysis of the graph, I comment out the
                                                                #   original code but I don't change the graph.
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|-------|
    sb5 = RN_REG_T5                                             #   R   |  SB5  |  SB6  |  SB3  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
    sb6 = RN_REG_T6                                             #       |       |       | dead* |       | input | input |       |       |
    sb7 = RN_REG_T0                                             #-------|-------|-------|-------|-------|-------|-------|-------|-------|
    RL[:] <= 0                                                  #  "0"  .       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                #   o>----------------------------->,   .       .       .       .       |
                                                                #       .       .       .       .   |   .       .       .       .       |
                                                                #   ,<------------------------------|------<o   .       .       .       |
                                                                #   |   .       .       .       .   |   .       .       .       .       |
    #     003 [0xffff : SB[7] = RL;;0xffff : RL = SB[1,2];]     #   &<------------------------------|--------------<o   .       .       |
    with apl_commands("read after write, 1st instance"):        #   |   .       .       .       .   |   .       .       .       .       |
        sb7[:] <= RL()                                          #   v   .       .       .       .   v   .       .       .       .       |
        RL [:] <= sb1() & sb2()                                 # 1 & 2 .       .       .       .   R*  . <~~ these updates are parallel|
                                                                #       .       .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       .       |
                                                                #       .       .   |   .       .       .       .       .       .       |
    #     004 [0xffff : SB[6] = RL;;0xffff : RL = SB[1];]       #   ,<--------------|----------------------<o   .       .       .       |
    with apl_commands("read after write, 2nd instance"):        #   |   .       .   |   .       .       .       .       .       .       |
        sb6[:] <= RL()                                          #   v   .       .   v   .       .       .       .       .       .       |
        RL [:] <= sb1()                                         #   1   .       .   R*  .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       .       |
                                                                # XOR<---------------------------------------------<o   .       .       |
                                                                #   |   .       .       .       .       .       .       .       .       |
    #     005 [0xffff : RL ^= SB[2];]                           #   v   .       .       .       .       .       .       .       .       |
    RL[:] ^= sb2()                                              # 2 ^ R .       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                #   o>----->.-------------->,   .       .       .       .       .       |
    #     006 [0xffff : SB[3,5] = RL;]                          #       .   |   .       .   |   .       .       .       .       .       |
    xffff = Belex.Mask("0xffff")                                #       .   v   .       .   v   .       .       .       .       .       |
#   xffff[sb3, sb5] <= RL()                                     #       .   R   .       .   R   .       .       .       .       .       |
    sb5[:] <= RL()                                              #       .       .       .       .       .       .       .       .       |
                                                                # x2222 <--<o  <~~ dead read after write; removed 1st cmd instr 007     |
                                                                #   5   .       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                # x2222 . <~~ reading 1111 of SRL is like reading 2222 of RL    .       |
                                                                #   o   .       .       .       .       .       .       .       .       |
                                                                #   |   .       . x1111 .       .       .       .       .       .       |
                                                                #   &<-------------<o   .       .       .       .       .       .       |
    #     007 [0x2222 : RL = SB[5];;0x1111 : RL = SB[6] & SRL;] #   |   .       .       .       .       .       .       .       .       |
    with apl_commands("compatible sections: 1st instance"):     #   v   .       .       .       .       .       .       .       .       |
        # RL["0x2222"] <= sb5()  # DEAD READ AFTER WRITE        # x1111 .       .       .       .       .       .       .       .       |
        RL["0x1111"] <= sb6() & SRL()  # uses old RL            # 6 & S .       .       .       .       .       .       .       .       |
        # only changed the code; graph reflects the original.   #       .       .       .       .       .       .       .       .       |
        # can't be laned with 006 due to spreader logic         #-------|-------|-------|-------|-------|-------|-------|-------|-------|
        # "attempt to update wr controls"                       #   R   |  SB5  |  SB6  |  SB3  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|-------|
                                                                # x1111 . <~~ reading 2222 of NRL is like reading 1111 of RL    .       |
                                                                #   o>--------------------->,   .       .       .       .       .       |
                                                                #   |   .       . x2222 .   |   .       .       .       .       .       |
                                                                #  OR<-------------<o   .   |   .       .       .       .       .       |
    #     008 [0x1111 : SB[3] = RL;;0x2222 : RL = SB[6] | NRL;] #   |   .       .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 2nd instance"):     #   v   .       .       .   v   .       .       .       .       .       |
#       sb3["0x1111"] <= RL()                                   # x2222 .       .       . x1111 .       .       .       .       .       |
        RL ["0x2222"] <= sb6() | NRL()  # uses old RL, th'fore  # 6 | N .       .       .   R   .       .       .       .       .       |
                                        # can't be moved up     #       .       .       .       .       .       .       .       .       |
                                                                # x2222 .       .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       .       |
    #     009 [0x2222 : SB[6] = RL;;0x1111 : RL = SB[5];]       #       .       .   |   .       .       .       .       .       .       |
    with apl_commands("compatible sections: 3rd instance"):     #       . x1111 .   v   .       .       .       .       .       .       |
        sb6["0x2222"] <= RL()                                   # x1111<---<o   . x2222 .       .       .       .       .       .       |
        RL ["0x1111"] <= sb5()                                  #   5   .       .   R   .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                # x1111 .       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       .       |
                                                                #   |   . x2222 .       .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       .       |
    #     010 [0x2222 : RL = SB[5] & NRL;]                      # x2222 .       .       .       .       .       .       .       .       |
    RL["0x2222"] <= sb5() & NRL()                               # 5 & N .       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                # x2222 . x4444 .       .       .       .       .       .       .       |
                                                                #   o   .   o   .       .       .       .       .       .       .       |
                                                                #    \  .  /    .       .       .       .       .       .       .       |
                                                                #     >===<     .       .       .       .       .       .       .       |
    #     011 [0x2222 : SB[5] = RL;;0x4444 : RL = SB[5];]       #    /  .  \    .       .       .       .       .       .       .       |
    with apl_commands("compatible sections: 4th instance"):     #   v   .   v   .       .       .       .       .       .       .       |
        sb5["0x2222"] <= RL()                                   # x4444 . x2222 .       .       .       .       .       .       .       |
        RL ["0x4444"] <= sb5()                                  #   5   .   R   .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                # x4444 . <~~ reading 2222 of SRL is like reading 4444 of RL    .       |
                                                                #   o   .       .       .       .       .       .       .       .       |
                                                                #   |   .       . x2222 .       .       .       .       .       .       |
                                                                #   &<-------------<o   .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       .       |
    #     012 [0x2222 : RL = SB[6] & SRL;]                      # x2222 .       .       .       .       .       .       .       .       |
    RL["0x2222"] <= sb6() & SRL()                               # 6 & S .       .       .       .       .       .       .       .       |
    # can't be moved up "attempt to update wr controls"         #       .       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB3  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|-------|
                                                                # x2222 . <~~ reading 4444 of NRL is like reading 2222 of RL    .       |
                                                                #   o>--------------------->,   .       .       .       .       .       |
                                                                #   |   .       . x4444 .   |   .       .       .       .       .       |
                                                                #  OR<-------------<o   .   |   .       .       .       .       .       |
    #     013 [0x2222 : SB[3] = RL;;0x4444 : RL = SB[6] | NRL;] #   |   .       .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 5th instance"):     #   v   .       .       .   v   .       .       .       .       .       |
#       sb3["0x2222"] <= RL()                                   # x4444 .       .       . x2222 .       .       .       .       .       |
        RL ["0x4444"] <= sb6() | NRL()                          # 6 | N .       .       .   R   .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                # x4444 .       .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       .       |
    #     014 [0x4444 : SB[6] = RL;;0x2222 : RL = SB[5];]       #       .       .   |   .       .       .       .       .       .       |
    with apl_commands("compatible sections: 6th instance"):     #       . x2222 .   v   .       .       .       .       .       .       |
        sb6["0x4444"] <= RL()                                   # x2222 <--<o   . x4444 .       .       .       .       .       .       |
        RL ["0x2222"] <= sb5()                                  #   5   .       .   R   .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                # x2222 .       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       .       |
                                                                #   |   . x4444 .       .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       .       |
    #     015 [0x4444 : RL = SB[5] & NRL;]                      # x4444 .       .       .       .       .       .       .       .       |
    RL["0x4444"] <= sb5() & NRL()                               # 5 & N .       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                # x4444 . x8888 .       .       .       .       .       .       .       |
                                                                #   o   .   o   .       .       .       .       .       .       .       |
                                                                #    \  .  /    .       .       .       .       .       .       .       |
                                                                #     >===<     .       .       .       .       .       .       .       |
    #     016 [0x4444 : SB[5] = RL;;0x8888 : RL = SB[5];]       #    /  .  \    .       .       .       .       .       .       .       |
    with apl_commands("compatible sections: 7th instance"):     #   v   .   v   .       .       .       .       .       .       .       |
        sb5["0x4444"] <= RL()                                   # x8888 . x4444 .       .       .       .       .       .       .       |
        RL ["0x8888"] <= sb5()                                  #   5   .   R   .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                # x8888 . <~~ reading 4444 of SRL is like reading 8888 of RL    .       |
                                                                #   o   .       .       .       .       .       .       .       .       |
                                                                #   |   .       . x4444 .       .       .       .       .       .       |
                                                                #   &<-------------<o   .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       .       |
    #     017 [0x4444 : RL = SB[6] & SRL;]                      # x4444 .       .       .       .       .       .       .       .       |
    RL["0x4444"] <= sb6() & SRL()                               # 6 & S .       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB3  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|-------|
                                                                # x4444 . <~~ reading 8888 of NRL is like reading 4444 of RL    .       |
                                                                #   o>--------------------->,   .       .       .       .       .       |
                                                                #   |   .       . x8888 .   |   .       .       .       .       .       |
                                                                #  OR<-------------<o   .   |   .       .       .       .       .       |
    #     018 [0x4444 : SB[3] = RL;;0x8888 : RL = SB[6] | NRL;] #   |   .       .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 8th instance"):     #   v   .       .       .   v   .       .       .       .       .       |
#       sb3["0x4444"] <= RL()                                   # x8888 .       .       . x4444 .       .       .       .       .       |
        RL ["0x8888"] <= sb6() | NRL()                          # 6 | N .       .       .   R   .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                # x8888 .       .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       .       |
    #     019 [0x8888 : SB[6] = RL;;0x4444 : RL = SB[5];]       #       .       .   |   .       .       .       .       .       .       |
    with apl_commands("compatible sections: 9th instance"):     #       . x4444 .   v   .       .       .       .       .       .       |
        sb6["0x8888"] <= RL()                                   # x4444 <--<o   . x8888 .       .       .       .       .       .       |
        RL ["0x4444"] <= sb5()                                  #   5   .       .   R   .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                # x4444 .       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       .       |
                                                                #   |   . x8888 .       .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       .       |
    #     020 [0x8888 : RL = SB[5] & NRL;]                      # x8888 .       .       .       .       .       .       .       .       |
    RL["0x8888"] <= sb5() & NRL()                               # 5 & N .       .       .       .       .       .       .       .       |
    # Can't merge above (needs new 4444:R).                     #       .       .       .       .       .       .       .       .       |
    # Can't merge below (wr controls)                           # x8888 .       .       .       .       .       .       .       .       |
                                                                #   o>----->,-------------->,   .       .       .       .       .       |
    #    021 [0x8888 : SB[3,5] = RL;]                           #       .   v   .       .   v   .       .       .       .       .       |
    eights = Belex.Mask("0x8888")                               #       . x8888 .       . x8888 .       .       .       .       .       |
#   eights[sb3, sb5] <= RL()                                    #       .   R   .       .   R   .       .       .       .       .       |
    with apl_commands("read after write; special instance"):    #       .       .       .       .               .       .       .       |
        sb5["0x8888"] <= RL() # eights[sb3, sb5] <= RL()        #       .       . <~~ 8888 killed in inst  021; all parts of 000f found.|
        RL["0x000F"] <= 0                                       #       . x000f . <~~ 2222 killed in instr 011; 4444 killed in instr 016|
                                                                #   ,<-----<o   . <~~ 1111 killed in instr  06. .       .       .       |
    # 22 has been merged into 21 and removed.                   #   |   .       .       .       .       .                       .       |
                                                                #   |   .       .       .       . x000f .       .       .       .       |
                                                                #   &<-----------------------------<o   . <~~ killed in instr 003       |
                                                                #   |   .       .       .       .       . <~~ replaced with literal "0" |
                                                                #   v   .       .       .       .       .       .       .       .       |
    #     022 [0x000f : RL = SB[5,7];]                          # x000f .       .       .       .       .       .       .       .       |
#   RL["0x000F"] <= sb5() & sb7()                               # 5 & 7 .       .       .       .       .       .       .       .       |
#   RL["0x000F"] <= 0  # The read of sb5 is unnecessary         #       .       .       .       .       .       .       .       .       |
                       # Rewrite 21 to have this read of "0"    #-------|-------|-------|-------|-------|-------|-------|-------|-------|
                       # in a laned read-after-write.           #   R   |  SB5  |  SB6  |  SB3  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                       # Graph continues to represent original. #-------|-------|-------|-------|-------|-------|-------|-------|-------|
                                                                # x0008 .       .       .       .       .       .       .       .       |
                                                                #   o>----------------------------------------------------->,   .       |
                                                                #       .       .       .       .       .       .       .   |   .       |
                                                                # x000f .       .       .       .       .       .       .   |   .       |
                                                                #   o   .       .       .       .       .       .       .   |   .       |
                                                                #   |   .       . x000f .       .       .       .       .   |   .       |
    #     023 [0x000f : RL |= SB[6];]                           #  OR<-------------<o   . <~~ overlap killed in 019     .   |   .       |
#   RL["0x000F"] |= sb6()                                       #   |   .       .       .       .       .       .       .   |   .       |
    with apl_commands("broadcast after read: 1st instance"):    #   v   .       .       .       .       .       .       .   v   .       |
        RL["0x000F"] |= sb6()                                   # x000f .       .       .       .       .       .       . x0008 .       |
        GL["0x0008"] <= RL()  # new RL, RL*                     # 6 | R .       .       .       .       .       .       .   R*  .       |
                                                                #       .       .   v~~ this kill is overwritten by     .       .       |
    #     024 [0x000f : SB[3,6] = RL;]                          # x000f .       .   v~~ the next two, one to x000e and other to x0001   |
#   x000f = Belex.Mask("0x000f")                                #   o>------------->,------>,   .       .       .       .       .       |
#   with apl_commands():                                        #       .       .   v   .   v   .       .       .       .       .       |
#       GL["0x0008"] <= RL()  # USES NEW RL; MOVED UP & MERGED  #       .       . x000f . x000f .       .       .       .       .       |
#       x000f[sb3, sb6] <= RL()  # IT DOESN'T MATTER!           #       .       .   R   .   R   .       .       .       .       .       |
#       sb6["0x000f"] <= RL()                                   #       .       .       .       .       .       .       .       .       |
# Split 000f into 000e and 0001. Merge 0001 portion into        # 0x007 .       .       .       .       .       .       .       .       |
# instr 27 below.                                               #   o>------------->,   .       .       .       .       .       .       |
                                                                #       .       .   |   .       . x0001 .       .       .       .       |
                                                                #   ,<--------------|--------------<o   . <~~ killed in instr 003       |
    #     027 [0x000e : SB[6] = NRL;;0x0001 : RL = SB[7];]      #   |   .       .   |   .       .       . <~~ replaced by literal "0"   |
    with apl_commands("compatible sections: 10th instance"):    #   v   .       .   v   .       .       .       .       .       .       |
        sb6["0x000e"] <= NRL()                                  # x0001 .       . x000e .       .       .       .       .       .       |
#       RL ["0x0001"] <= sb7()                                  #   7   .       .   N   .       .       .       .       .       .       |
        RL ["0x0001"] <= 0                                      #       .       .       .       .       .       .       .       .       |
#       sb6["0x0001"] <= RL()                                   # x0001 .       .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       .       |
    #     028 [0x0001 : SB[6] = RL;;0xffff : RL = SB[4];]       #       .       .   v   .       .       .       .       .       .       |
    with apl_commands("read after write: 2nd instance"):        #       .       . x0001 .       .       .       .       .       .       |
        sb6["0x0001"] <= RL()                                   #       .       .   R   .       .       .       .       .       .       |
# Can't merge this downward (attempt to update wr controls)     #       .       .       .       .       .       .       . x00f0 .       |
                                                                #   ,<-----------------------------------------------------<o   .       |
                                                                #   |   . x00f0 .       .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       .       |
    #     030 [0x00f0 : RL = SB[5,7];]                          #   |   .       .       .       .   ,<---------------------<o   .       |
    with apl_commands():                                        #   v   .       .       .       .   |   .       .       .       .       |
        RL["0x00f0"] <= sb5() & GL()  # sb7()                   # x00f0 .       .       .       .   v   .       .       .       .       |
        sb7[:]       <= GL()                                    # 5 & G .       .       .       .   G   .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB3  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|-------|
                                                                # x00f0 .       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       .       |
                                                                #   |   .       . x00f0 .       .       .       .       .       .       |
                                                                #  OR<-------------<o   .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       .       |
    #     031 [0x00f0 : RL |= SB[6];]                           # x00f0 .       .       .       .       .       .       .       .       |
#   RL["0x00f0"] |= sb6()                                       # 6 | R .       .       .       .       .       .       .       .       |
    with apl_commands():                                        #       .       .       .       .       .       .       .       .       |
        RL["0x00f0"] |= sb6()                                   # x0080 .       .       .       .       .       .       .       .       |
        GL["0x0080"] <= RL()                                    #   o>----------------------------------------------------->,   .       |
                                                                #       .       .       .       .       .       .       .   |   .       |
    #     032 [0x00f0 : SB[3,6] = RL;]                          # x00f0 .       .       .       .       .       .       .   |   .       |
#   x00f0 = Belex.Mask("0x00f0")                                #   o>------------->,------>,   .       .       .       .   |   .       |
#   with apl_commands():                                        #       .       .   v   .   v   .       .       .       .   v   .       |
#       GL["0x0080"] <= RL()  # MOVE UP AND MERGE UPWARD.       #       .       . x00f0 . x00f0 .       .       .       . x0080 .       |
#       x00f0[sb3, sb6] <= RL()                                 #       .       .   R   .   R   .       .       .       .   R   .       |
#       sb6["0x00f0"] <= RL()                                   #       .       .       .       .       .       .       .       .       |
# Split 00f0 into 00e0 and 0010, merge downward.                # x0070 .       .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       .       |
                                                                #       .       .   |   .       . x0010 .       .       .       .       |
                                                                #   ,<--------------|--------------<o   .       .       .       .       |
    #     035 [0x00e0 : SB[6] = NRL;;0x0010 : RL = SB[7];]      #   |   .       .   |   .       .       .       .       .       .       |
    with apl_commands("compatible sections: 11th instance"):    #   v   .       .   v   .       .       .       .       .       .       |
        sb6["0x00e0"] <= NRL()                                  # x0010 .       . x00e0 .       .       .       .       .       .       |
        RL ["0x0010"] <= sb7()                                  #   7   .       .   N   .       .       .       .       .       .       |
#       sb6["0x0010"] <= RL()                                   #       .       .       .       .       .       .       .       .       |
                                                                # x0010 .       .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       .       |
    #     036 [0x0010 : SB[6] = RL;;0xffff : RL = SB[4];]       #       .       .   v   .       .       .       .       .       .       |
    with apl_commands("read after write: 4th instance"):        #       .       . x0100 .       .       .       .       .       .       |
        sb6["0x0010"] <= RL()                                   #       .       .   R   .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       . x0f00 .       |
                                                                #   ,<-----------------------------------------------------<o   .       |
                                                                #   |   . x0f00 .       .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .   ,<---------------------<o   .       |
    #     038 [0x0f00 : RL = SB[5,7];]  # OPPORTUNITY?          #   |   .       .       .       .   |   .       .       .       .       |
    with apl_commands():                                        #   v   .       .       .       .   |   .       .       .       .       |
        RL["0x0f00"] <= sb5() & GL()  # sb7()                   # x0f00 .       .       .       .   v   .       .       .       .       |
        sb7[:]       <= GL()  # RL()                            # 5 & G .       .       .       .   G   .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB3  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|-------|
                                                                # x0f00 .       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       .       |
                                                                #   |   .       . x0f00 .       .       .       .       .       .       |
                                                                #  OR<-------------<o   .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       .       |
    #     039 [0x0f00 : RL |= SB[6];]                           # x0f00 .       .       .       .       .       .       .       .       |
#   RL["0x0f00"] |= sb6()                                       # 6 | R .       .       .       .       .       .       .       .       |
    with apl_commands():                                        #       .       .       .       .       .       .       .       .       |
        RL["0x0f00"] |= sb6()                                   # x0800 .       .       .       .       .       .       .       .       |
        GL["0x0800"] <= RL()                                    #   o>----------------------------------------------------->,   .       |
                                                                #       .       .       .       .       .       .       .   |   .       |
    #     040 [0x0f00 : SB[3,6] = RL;]                          # x0f00 .       .       .       .       .       .       .   |   .       |
#   x0f00 = Belex.Mask("0x0f00")                                #   o>------------->,------>,   .       .       .       .   |   .       |
#   with apl_commands():                                        #       .       .   v   .   v   .       .       .       .   v   .       |
#       GL["0x0800"] <= RL()                                    #       .       . x0f00 . x0f00 .       .       .       . x0800 .       |
#       x0f00[sb3, sb6] <= RL()                                 #       .       .   R   .   R   .       .       .       .   R   .       |
#       sb6["0x0f00"] <= RL()                                   #       .       .       .       .       .       .       .       .       |
                                                                # x0700 .       .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       .       |
                                                                #       .       .   |   .       . x0100 .       .       .       .       |
                                                                #   ,<--------------|--------------<o   .       .       .       .       |
    #     043 [0x0e00 : SB[6] = NRL;;0x0100 : RL = SB[7];]      #   |   .       .   |   .       .       .       .       .       .       |
    with apl_commands("compatible sections: 12th instance"):    #   v   .       .   v   .       .       .       .       .       .       |
        sb6["0x0e00"] <= NRL()                                  # x0100 .       . x0e00 .       .       .       .       .       .       |
        RL ["0x0100"] <= sb7()                                  #   7   .       .   N   .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                # x0100 .       .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       .       |
    #     044 [0x0100 : SB[6] = RL;;0xffff : RL = SB[4];]       #       .       .   v   .       .       .       .       .       .       |
    with apl_commands("read after write: 5th instance"):        #       .       . x0100 .       .       .       .       .       .       |
        sb6["0x0100"] <= RL()                                   #       .       .   R   .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       . xf000 .       |
                                                                #   ,<-----------------------------------------------------<o   .       |
                                                                #   |   . xf000 .       .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       .       |
    #     046 [0xf000 : RL = SB[5,7];]                          #   |   .       .       .       .       .       .       .       .       |
    # UNREALISTIC TO REPLACE THE ALIAS OF GL FOR SB7            #   v   .       .       .       .       .       .       .       .       |
    with apl_commands():                                        # xf000 .       .       .       .       .       .       .       .       |
        RL["0xf000"] <= sb5() & GL()  # sb7()                   # 5 & G .       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB3  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|-------|
                                                                # xf000 .       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       .       |
                                                                #   |   .       . xf000 .       .       .       .       .       .       |
                                                                #  OR<-------------<o   .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       .       |
    #     047 [0xf000 : RL |= SB[6];]                           # xf000 .       .       .       .       .       .       .       .       |
    RL["0xf000"] |= sb6()                                       # 6 | R .       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                # xf000 .       .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       .       |
                                                                #       .       .   v   .       .       .       .       .       .       |
    #     048 [0xf000 : SB[6] = RL;]                            #       .       . xf000 . <~~ dead write.       .       .       .       |
#   sb6["0xf000"] <= RL()                                       #       .       .   R   .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                # x7000 .       .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       .       |
    #     051 [0xe000 : SB[6] = NRL;;0x1000 : RL = SB[7];]      #       .       .   v   .       .       .       .       .       .       |
#   with apl_commands("compatible sections: 13th instance"):    #       .       . xe000 .       .       .       .       .       .       |
#       sb6["0xe000"] <= NRL()                                  #       .       .   N   .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                #   ,<-------------------------------------<o   .       .       .       |
                                                                #   |   .       .       .       .       .       .       . x1000 .       |
    #     052 [0x1000 : SB[6] = RL;;0xffff : RL = SB[1];]       #   |   .       .   ,<-------------------------------------<o   .       |
    with apl_commands("read after write: 5th instance"):        #   |   .       .   v   .       .       .       .       .       .       |
        sb6["0x1000"] <= GL()  # RL()                           #   v   .       . x1000 .       .       .       .       .       .       |
        RL [   ::   ] <= sb1()                                  #   1   .       .   G   .       .       .       .       .       .       |
        sb6["0xe000"] <= NRL()                                  #       .       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       .       |
                                                                # XOR<-------------<o   .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       .       |
    #     053 [0xffff : RL ^= SB[6];]                           #   v   .       .       .       .       .       .       .       .       |
    RL[::] ^= sb6()                                             # 6 ^ R .       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       .       |
                                                                # XOR<---------------------------------------------<o   .       .       |
    #     054 [0xffff : SB[3] = RL;;0xffff : RL = SB[2];]       #   |   .       .       .       .       .       .       .       .       |
    with apl_commands("read after write: 6th instance"):        #   v   .       .       .       .       .       .       .       .       |
        RL [:] ^= sb2()                                         # 2 ^ R .       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       .       |
                                                                #   o>------------------------------------------------------------->,   |
                                                                #       .       .       .       .       .       .       .       .   |   |
                                                                #       .       .       .       .       .       .       .       .   v   |
    out[:] <= RL()                                              #       .       .       .       .       .       .       .       .   R   |



@belex_apl
def laning_laboratory_009(Belex, out: VR, sb1: VR, sb2: VR):
                                                                #                     KILLS BY INSTRUCTION NUMBER
                                                                # KILLS have arrows pointing IN to a node; GENS have arrows pointing out.
                                                                # Lack of a section mask means "all sections."
                                                                # In a KILL notation, the node indicates the value deposited.
                                                                # R* means that the READ (loading RL) occurs after a WRITE (loading SB).
                                                                # R* means that BROADCAST (loading GL) occurs after a READ (loading RL).
                                                                # Nodes on the same horizontal line are effected in parallel lanes.
                                                                # To figure out liveness of sections, do mask math in your head; look
                                                                #   upward in a column to find the bottom-most KILL of certain sections.
                                                                # The graph always contains dependencies from the original code. When I
                                                                #   make changes revealed by analysis of the graph, I comment out the
                                                                #   original code but I don't change the graph.
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
    sb5 = RN_REG_T5                                             #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
    sb6 = RN_REG_T6                                             #       |       |       |       | input | input |       |       |
    sb7 = RN_REG_T0                                             #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
    RL[:] <= 0                                                  #  "0"  . <~~ read      .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                #   o>--------------------->,   . <~~ write     .       .       |
                                                                #       .       .       .   |   .       .       .       .       |
                                                                #   ,<----------------------|------<o   . <~~ read      .       |
                                                                #   |   .       .       .   |   .       .       .       .       |
                                                                #   v   .       .       .   |   .       .       .       .       |
    #     003 [0xffff : SB[7] = RL;;0xffff : RL = SB[1,2];]     #   &<----------------------|--------------<o   . <~~ read      |
    with apl_commands("read after write, 1st instance"):        #   |   .       .       .   |   .       .       .       .       |
        sb7[:] <= RL()                                          #   v   .       .       .   v   .       .       .       .       |
        RL [:] <= sb1() & sb2()                                 # 1 & 2 .       .       .   R*  . <~~ these updates are parallel|
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                #   o>------------->,   . <~~ write     .       .       .       |
                                                                #       .       .   |   .       .       .       .       .       |
    #     004 [0xffff : SB[6] = RL;;0xffff : RL = SB[1];]       #   ,<--------------|--------------<o   . <~~ read      .       |
    with apl_commands("read after write, 2nd instance"):        #   |   .       .   |   .       .       .       .       .       |
        sb6[:] <= RL()                                          #   v   .       .   v   .       .       .       .       .       |
        RL [:] <= sb1()                                         #   1   .       .   R*  .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
                                                                # XOR<-------------------------------------<o   .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
    #     005 [0xffff : RL ^= SB[2];]                           #   v   .       .       .       .       .       .       .       |
    RL[:] ^= sb2()                                              # 2 ^ R .       .       .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
    #     006 [0xffff : SB[3,5] = RL;]                          #       .       .       .       .       .       .       .       |
    sb5[:] <= RL()                                              #   o>----->R   .       .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x2222 . <~~ reading 1111 of SRL is like reading 2222 of RL    |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       . x1111 .       .       .       .       .       |
                                                                #   &<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
    #     007 [0x2222 : RL = SB[5];;0x1111 : RL = SB[6] & SRL;] #   v   .       .       .       .       .       .       .       |
    with apl_commands("compatible sections: 1st instance"):     # x1111 .       .       .       .       .       .       .       |
        RL["0x1111"] <= sb6() & SRL()  # uses old RL            # 6 & S .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
                                                                # x1111 . <~~ reading 2222 of NRL is like reading 1111 of RL    |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       . x2222 .       .       .       .       .       |
                                                                #  OR<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
    #     008 [0x1111 : SB[3] = RL;;0x2222 : RL = SB[6] | NRL;] #   v   .       .       .       .       .       .       .       |
    with apl_commands("compatible sections: 2nd instance"):     # x2222 .       .       .       .       .       .       .       |
        RL ["0x2222"] <= sb6() | NRL()  # uses old RL, th'fore  # 6 | N .       .       .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x2222 .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       |
    #     009 [0x2222 : SB[6] = RL;;0x1111 : RL = SB[5];]       #       .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 3rd instance"):     #       . x1111 .   v   .       .       .       .       .       |
        sb6["0x2222"] <= RL()                                   # x1111<---<o   . x2222 .       .       .       .       .       |
        RL ["0x1111"] <= sb5()                                  #   5   .       .   R   .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x1111 .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   . x2222 .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
    #     010 [0x2222 : RL = SB[5] & NRL;]                      # x2222 .       .       .       .       .       .       .       |
    RL["0x2222"] <= sb5() & NRL()                               # 5 & N .       .       .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x2222 . x4444 .       .       .       .       .       .       |
                                                                #   o   .   o   .       .       .       .       .       .       |
                                                                #    \  .  /    .       .       .       .       .       .       |
                                                                #     >===<     .       .       .       .       .       .       |
    #     011 [0x2222 : SB[5] = RL;;0x4444 : RL = SB[5];]       #    /  .  \    .       .       .       .       .       .       |
    with apl_commands("compatible sections: 4th instance"):     #   v   .   v   .       .       .       .       .       .       |
        sb5["0x2222"] <= RL()                                   # x4444 . x2222 .       .       .       .       .       .       |
        RL ["0x4444"] <= sb5()                                  #   5   .   R   .       .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x4444 . <~~ reading 2222 of SRL is like reading 4444 of RL    |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       . x2222 .       .       .       .       .       |
                                                                #   &<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
    #     012 [0x2222 : RL = SB[6] & SRL;]                      # x2222 .       .       .       .       .       .       .       |
    RL["0x2222"] <= sb6() & SRL()                               # 6 & S .       .       .       .       .       .       .       |
    # can't be moved up "attempt to update wr controls"         #       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
                                                                # x2222 . <~~ reading 4444 of NRL is like reading 2222 of RL    |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       . x4444 .       .       .       .       .       |
                                                                #  OR<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
    #     013 [0x2222 : SB[3] = RL;;0x4444 : RL = SB[6] | NRL;] #   v   .       .       .       .       .       .       .       |
    with apl_commands("compatible sections: 5th instance"):     # x4444 .       .       .       .       .       .       .       |
        RL ["0x4444"] <= sb6() | NRL()                          # 6 | N .       .       .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x4444 .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       |
    #     014 [0x4444 : SB[6] = RL;;0x2222 : RL = SB[5];]       #       .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 6th instance"):     #       . x2222 .   v   .       .       .       .       .       |
        sb6["0x4444"] <= RL()                                   # x2222 <--<o   . x4444 .       .       .       .       .       |
        RL ["0x2222"] <= sb5()                                  #   5   .       .   R   .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x2222 . <~~ reading 4444 of NRL is like reading 2222 of RL    |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   . x4444 .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
    #     015 [0x4444 : RL = SB[5] & NRL;]                      # x4444 .       .       .       .       .       .       .       |
    RL["0x4444"] <= sb5() & NRL()                               # 5 & N .       .       .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x4444 . x8888 .       .       .       .       .       .       |
                                                                #   o   .   o   .       .       .       .       .       .       |
                                                                #    \  .  /    .       .       .       .       .       .       |
                                                                #     >===<     .       .       .       .       .       .       |
    #     016 [0x4444 : SB[5] = RL;;0x8888 : RL = SB[5];]       #    /  .  \    .       .       .       .       .       .       |
    with apl_commands("compatible sections: 7th instance"):     #   v   .   v   .       .       .       .       .       .       |
        sb5["0x4444"] <= RL()                                   # x8888 . x4444 .       .       .       .       .       .       |
        RL ["0x8888"] <= sb5()                                  #   5   .   R   .       .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x8888 . <~~ reading 4444 of SRL is like reading 8888 of RL    |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       . x4444 .       .       .       .       .       |
                                                                #   &<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
    #     017 [0x4444 : RL = SB[6] & SRL;]                      # x4444 .       .       .       .       .       .       .       |
    RL["0x4444"] <= sb6() & SRL()                               # 6 & S .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
                                                                # x4444 . <~~ reading 8888 of NRL is like reading 4444 of RL    |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       . x8888 .       .       .       .       .       |
                                                                #  OR<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
    #     018 [0x4444 : SB[3] = RL;;0x8888 : RL = SB[6] | NRL;] #   v   .       .       .       .       .       .       .       |
    with apl_commands("compatible sections: 8th instance"):     # x8888 .       .       .       .       .       .       .       |
        RL ["0x8888"] <= sb6() | NRL()                          # 6 | N .       .       .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x8888 .       .       .       .       .       .       .       |
                                                                #   o>------------->,   . <~~ write     .       .       .       |
                                                                #       .       .   |   .       .       .       .       .       |
                                                                #       . x4444 .   |   .       .       .       .       .       |
                                                                #   ,<-----<o   .   |   . <~~ read      .       .       .       |
    #     019 [0x8888 : SB[6] = RL;;0x4444 : RL = SB[5];]       #   |   .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 9th instance"):     #   v   .       .   v   .       .       .       .       .       |
        sb6["0x8888"] <= RL()                                   # x4444 .       . x8888 .       .       .       .       .       |
        RL ["0x4444"] <= sb5()                                  #   5   .       .   R   .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x4444 .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   . x8888 .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
    #     020 [0x8888 : RL = SB[5] & NRL;]                      # x8888 .       .       .       .       .       .       .       |
    RL["0x8888"] <= sb5() & NRL()                               # 5 & N .       .       .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x8888 .       .       .       .       .       .       .       |
                                                                #   o>----->,   . <~~ write     .       .       .       .       |
                                                                #       .   |   .       .       .       .       .       .       |
                                                                #  "0"  .   |   . <~~ read      .       .       .       .       |
    #    021 [0x8888 : SB[3,5] = RL;]                           #   |   .   |   .       .       .       .       .       .       |
    with apl_commands("read after write; special instance"):    #   v   .   v   .       .       .       .       .       .       |
        sb5["0x8888"] <= RL()                                   # x000f . x8888 .       .       .       .       .       .       |
        RL["0x000F"] <= 0                                       #  "0"  .   R   .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
                                                                # x000f .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       . x000f .       .       .       .       .       |
                                                                #  OR<-------------<o   . <~~ read      .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
                                                                # x0008 .       . ~~> broadcast .       .       .       .       |
                                                                #   o>--------------------------------------------->,   .       |
    #     023 [0x000f : RL |= SB[6];]                           #   |   .       .       .       .       .       .   |   .       |
    with apl_commands("broadcast after read: 1st instance"):    #   v   .       .       .       .       .       .   v   .       |
        RL["0x000F"] |= sb6()                                   # x000f .       .       .       .       .       . x0008 .       |
        GL["0x0008"] <= RL()  # new RL, RL*                     # 6 | R .       .       .       .       .       .   R*  .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x0007 . <~~ reading 000e of NRL is like reading 0007 of RL    |
                                                                #   o>------------->,   .       .       .       .       .       |
                                                                #       .       .   |   .       .       .       .       .       |
                                                                #  "0"  . <~~ read  |   .       .       .       .       .       |
    #     027 [0x000e : SB[6] = NRL;;0x0001 : RL = SB[7];]      #   |   .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 10th instance"):    #   v   .       .   v   .       .       .       .       .       |
        sb6["0x000e"] <= NRL()                                  # x0001 .       . x000e . <~~ write     .       .       .       |
        RL ["0x0001"] <= 0                                      #   7   .       .   N   .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
    #     028 [0x0001 : SB[6] = RL;;0xffff : RL = SB[4];]       # x0001 .       .       .       .       .       .       .       |
    with apl_commands():                                        #   o>----------> x0001 .       .       .       .       .       |
        sb6["0x0001"] <= RL()                                   #       .       .   R   .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       . x00f0 .       |
                                                                #   ,<---------------------------------------------<o   .       |
                                                                #   v   . x00f0 .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       |
    #     030 [0x00f0 : RL = SB[5,7];]                          #   |   .       .       .       .       .       .       .       |
    with apl_commands("compatible sections; 11th instance?"):   #   v   .       .       .       .       .       .       .       |
        RL["0x00f0"] <= sb5() & GL()  # sb7()                   # x00f0 .       .       .       .       .       .       .       |
        sb7[:]       <= GL()                                    # 5 & G .       .       .   G <--------------------<o   .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
                                                                # x00f0 .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       . x00f0 .       .       .       .       .       |
                                                                #  OR<-------------<o   . <~~ read      .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
                                                                # x0080 .       . ~~> broadcast .       .       .       .       |
                                                                #   o>--------------------------------------------->,   .       |
    #     031 [0x00f0 : RL |= SB[6];]                           #   |   .       .       .       .       .       .   |   .       |
    with apl_commands("broadcast after read"):                  #   v   .       .       .       .       .       .   v   .       |
        RL["0x00f0"] |= sb6()                                   # x00f0 .       .       .       .       .       . x0080 .       |
        GL["0x0080"] <= RL()                                    # 6 | R .       .       .       .       .       .   R*  .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x0070 . <~~ write     .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       |
                                                                #       .       .   |   . x0010 .       .       .       .       |
                                                                #   ,<--------------|------<o   . <~~ read      .       .       |
    #     035 [0x00e0 : SB[6] = NRL;;0x0010 : RL = SB[7];]      #   |   .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 11th instance"):    #   v   .       .   v   .       .       .       .       .       |
        sb6["0x00e0"] <= NRL()                                  # x0010 .       . x00e0 .       .       .       .       .       |
        RL ["0x0010"] <= sb7()                                  #   7   .       .   N   .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
    #     036 [0x0010 : SB[6] = RL;;0xffff : RL = SB[4];]       # x0010 .       .       .       .       .       .       .       |
    with apl_commands():                                        #   o>----------> x0100 .       .       .       .       .       |
        sb6["0x0010"] <= RL()                                   #       .       .   R   .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       . x0f00 .       |
                                                                #   ,<---------------------------------------------<o   .       |
                                                                #   |   . x0f00 .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       |
    #     038 [0x0f00 : RL = SB[5,7];]  # OPPORTUNITY?          #   |   .       .       .       .       .       .       .       |
    with apl_commands():                                        #   v   .       .       .       .       .       .       .       |
        RL["0x0f00"] <= sb5() & GL()  # sb7()                   # x0f00 .       .       .       .       .       .       .       |
        sb7[:]       <= GL()  # RL()                            # 5 & G .       .       .   G <--------------------<o   .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
                                                                # x0f00 .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       . x0f00 .       .       .       .       .       |
                                                                #  OR<-------------<o   . <~~ read      .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
                                                                # x0800 .       . ~~> broadcast .       .       .       .       |
                                                                #   o>--------------------------------------------->,00 .       |
    #     039 [0x0f00 : RL |= SB[6];]                           #   |   .       .       .       .       .       .   |   .       |
    with apl_commands():                                        #   v   .       .       .       .       .       .   v   .       |
        RL["0x0f00"] |= sb6()                                   # x00f0 .       .       .       .       .       . x0800 .       |
        GL["0x0800"] <= RL()                                    # 6 | R .       .       .               .       .   R   .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x0700 . <~~ write     .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       |
                                                                #       .       .   |   . x0100 .       .       .       .       |
                                                                #   ,<--------------|------<o   . <~~ read      .       .       |
    #     043 [0x0e00 : SB[6] = NRL;;0x0100 : RL = SB[7];]      #   |   .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 12th instance"):    #   v   .       .   v   .       .       .       .       .       |
        sb6["0x0e00"] <= NRL()                                  # x0100 .       . x0e00 .       .       .       .       .       |
        RL ["0x0100"] <= sb7()                                  #   7   .       .   N   .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
    #     044 [0x0100 : SB[6] = RL;;0xffff : RL = SB[4];]       # x0100 .       .       .       .       .       .       .       |
    with apl_commands("read after write: 5th instance"):        #   o>----------> x0100 .       .       .       .       .       |
        sb6["0x0100"] <= RL()                                   #       .       .   R   .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       . xf000 .       |
                                                                #   ,<---------------------------------------------<o   .       |
                                                                #   |   . xf000 .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       |
    #     046 [0xf000 : RL = SB[5,7];]                          #   |   .       .       .       .       .       .       .       |
    # UNREALISTIC TO REPLACE THE ALIAS OF GL FOR SB7            #   v   .       .       .       .       .       .       .       |
    with apl_commands():                                        # xf000 .       .       .       .       .       .       .       |
        RL["0xf000"] <= sb5() & GL()  # sb7()                   # 5 & G .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
                                                                # xf000 .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       . xf000 .       .       .       .       .       |
                                                                #  OR<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
    #     047 [0xf000 : RL |= SB[6];]                           # xf000 .       .       .       .       .       .       .       |
    RL["0xf000"] |= sb6()                                       # 6 | R .       .       .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x7000 .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       |
                                                                #       .       .   |   .       .       .       .       .       |
                                                                #       .       .   v   .       .       .       .       .       |
                                                                #       .       . xe000 .       .       .       .       .       |
    #     052 [0x1000 : SB[6] = RL;;0xffff : RL = SB[1];]       #       .       .   |   .       .       .       .       .       |
    with apl_commands("read after write: 5th instance"):        #   ,<--------------|--------------<o   .       .       .       |
        sb6["0x1000"] <= GL()  # RL()                           #   |   .       .   v   .       .       .       . x1000 .       |
        RL [   ::   ] <= sb1()                                  #   v   .       . x1000 <--------------------------<o   .       |
        sb6["0xe000"] <= NRL()                                  #   1   .       .   G   .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                # XOR<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
    #     053 [0xffff : RL ^= SB[6];]                           #   v   .       .       .       .       .       .       .       |
    RL[::] ^= sb6()                                             # 6 ^ R .       .       .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                # XOR<-------------------------------------<o   .       .       |
    #     054 [0xffff : SB[3] = RL;;0xffff : RL = SB[2];]       #   |   .       .       .       .       .       .       .       |
    with apl_commands("read after write: 6th instance"):        #   v   .       .       .       .       .       .       .       |
        RL [:] ^= sb2()                                         # 2 ^ R .       .       .       .       .       .       .       |
                                                                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |
                                                                #       .       .       .       .       .       .       .       |
    out[:] <= RL()                                              #   o>----------------------------------------------------> R   |
                                                                #       .       .       .       .       .       .       .       |
                                                                #---------------------------------------------------------------|


@belex_apl
def laning_laboratory_010(Belex, out: VR, sb1: VR, sb2: VR):
                                                                #                     KILLS BY INSTRUCTION NUMBER
                                                                # KILLS have arrows pointing IN to a node; GENS have arrows pointing out.
                                                                # Lack of a section mask means "all sections."
                                                                # In a KILL notation, the node indicates the value deposited.
                                                                # R* means that the READ (loading RL) occurs after a WRITE (loading SB).
                                                                # R* means that BROADCAST (loading GL) occurs after a READ (loading RL).
                                                                # Nodes on the same horizontal line are effected in parallel lanes.
                                                                # To figure out liveness of sections, do mask math in your head; look
                                                                #   upward in a column to find the bottom-most KILL of certain sections.
                                                                # The graph always contains dependencies from the original code. When I
                                                                #   make changes revealed by analysis of the graph, I comment out the
                                                                #   original code but I don't change the graph.
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
    sb5 = RN_REG_T5                                             #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
    sb6 = RN_REG_T6                                             #       |       |       |       | input | input |       |       |
    sb7 = RN_REG_T0                                             #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
    RL[:] <= 0                                                  #  "0"  . <~~ read      .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #   o>--------------------->,   . <~~ write     .       .       |
                                                                #       .       .       .   |   .       .       .       .       |
                                                                #   ,<----------------------|------<o   . <~~ read      .       |
                                                                #   |   .       .       .   |   .       .       .       .       |
                                                                #   v   .       .       .   |   .       .       .       .       |
    #     003 [0xffff : SB[7] = RL;;0xffff : RL = SB[1,2];]     #   &<----------------------|--------------<o   . <~~ read      |
    with apl_commands("read after write, 1st instance"):        #   |   .       .       .   |   .       .       .       .       |
        sb7[:] <= RL()                                          #   v   .       .       .   v   .       .       .       .       |
        RL [:] <= sb1() & sb2()                                 # 1 & 2 .       .       .   R*  . <~~ these updates are parallel|
                                                                #       .       .       .       .       .       .       .       |
                                                                #   o>------------->,   . <~~ write     .       .       .       |
                                                                #       .       .   |   .       .       .       .       .       |
    #     004 [0xffff : SB[6] = RL;;0xffff : RL = SB[1];]       #   ,<--------------|--------------<o   . <~~ read      .       |
    with apl_commands("read after write, 2nd instance"):        #   |   .       .   |   .       .       .       .       .       |
        sb6[:] <= RL()                                          #   v   .       .   v   .       .       .       .       .       |
        RL [:] <= sb1()                                         #   1   .       .   R*  .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
                                                                # XOR<-------------------------------------<o   .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
    #     005 [0xffff : RL ^= SB[2];]                           #   v   .       .       .       .       .       .       .       |
    RL[:] ^= sb2()                                              # 2 ^ R .       .       .       .       .       .       .       |
    #     006 [0xffff : SB[3,5] = RL;]                          #       .       .       .       .       .       .       .       |
    sb5[:] <= RL()                                              #   o>----->R   .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x2222 . <~~ reading 1111 of SRL is like reading 2222 of RL    |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       . x1111 .       .       .       .       .       |
                                                                #   &<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
    #     007 [0x2222 : RL = SB[5];;0x1111 : RL = SB[6] & SRL;] #   v   .       .       .       .       .       .       .       |
    with apl_commands("compatible sections: 1st instance"):     # x1111 .       .       .       .       .       .       .       |
        RL["0x1111"] <= sb6() & SRL()  # uses old RL            # 6 & S .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
                                                                # x1111 . <~~ reading 2222 of NRL is like reading 1111 of RL    |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       . x2222 .       .       .       .       .       |
                                                                #  OR<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
    #     008 [0x1111 : SB[3] = RL;;0x2222 : RL = SB[6] | NRL;] #   v   .       .       .       .       .       .       .       |
    with apl_commands("compatible sections: 2nd instance"):     # x2222 .       .       .       .       .       .       .       |
        RL ["0x2222"] <= sb6() | NRL()  # uses old RL, th'fore  # 6 | N .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x2222 .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       |
    #     009 [0x2222 : SB[6] = RL;;0x1111 : RL = SB[5];]       #       .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 3rd instance"):     #       . x1111 .   v   .       .       .       .       .       |
        sb6["0x2222"] <= RL()                                   # x1111<---<o   . x2222 .       .       .       .       .       |
        RL ["0x1111"] <= sb5()                                  #   5   .       .   R   .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x1111 .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   . x2222 .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
    #     010 [0x2222 : RL = SB[5] & NRL;]                      # x2222 .       .       .       .       .       .       .       |
    RL["0x2222"] <= sb5() & NRL()                               # 5 & N .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x2222 . x4444 .       .       .       .       .       .       |
                                                                #   o   .   o   .       .       .       .       .       .       |
                                                                #    \  .  /    .       .       .       .       .       .       |
                                                                #     >===<     .       .       .       .       .       .       |
    #     011 [0x2222 : SB[5] = RL;;0x4444 : RL = SB[5];]       #    /  .  \    .       .       .       .       .       .       |
    with apl_commands("compatible sections: 4th instance"):     #   v   .   v   .       .       .       .       .       .       |
        sb5["0x2222"] <= RL()                                   # x4444 . x2222 .       .       .       .       .       .       |
        RL ["0x4444"] <= sb5()                                  #   5   .   R   .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x4444 . <~~ reading 2222 of SRL is like reading 4444 of RL    |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       . x2222 .       .       .       .       .       |
                                                                #   &<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
    #     012 [0x2222 : RL = SB[6] & SRL;]                      # x2222 .       .       .       .       .       .       .       |
    RL["0x2222"] <= sb6() & SRL()                               # 6 & S .       .       .       .       .       .       .       |
    # can't be moved up "attempt to update wr controls"         #       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
                                                                # x2222 . <~~ reading 4444 of NRL is like reading 2222 of RL    |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       . x4444 .       .       .       .       .       |
                                                                #  OR<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
    #     013 [0x2222 : SB[3] = RL;;0x4444 : RL = SB[6] | NRL;] #   v   .       .       .       .       .       .       .       |
    with apl_commands("compatible sections: 5th instance"):     # x4444 .       .       .       .       .       .       .       |
        RL ["0x4444"] <= sb6() | NRL()                          # 6 | N .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x4444 .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       |
    #     014 [0x4444 : SB[6] = RL;;0x2222 : RL = SB[5];]       #       .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 6th instance"):     #       . x2222 .   v   .       .       .       .       .       |
        sb6["0x4444"] <= RL()                                   # x2222 <--<o   . x4444 .       .       .       .       .       |
        RL ["0x2222"] <= sb5()                                  #   5   .       .   R   .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x2222 . <~~ reading 4444 of NRL is like reading 2222 of RL    |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   . x4444 .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
    #     015 [0x4444 : RL = SB[5] & NRL;]                      # x4444 .       .       .       .       .       .       .       |
    RL["0x4444"] <= sb5() & NRL()                               # 5 & N .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x4444 . x8888 .       .       .       .       .       .       |
                                                                #   o   .   o   .       .       .       .       .       .       |
                                                                #    \  .  /    .       .       .       .       .       .       |
                                                                #     >===<     .       .       .       .       .       .       |
    #     016 [0x4444 : SB[5] = RL;;0x8888 : RL = SB[5];]       #    /  .  \    .       .       .       .       .       .       |
    with apl_commands("compatible sections: 7th instance"):     #   v   .   v   .       .       .       .       .       .       |
        sb5["0x4444"] <= RL()                                   # x8888 . x4444 .       .       .       .       .       .       |
        RL ["0x8888"] <= sb5()                                  #   5   .   R   .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x8888 . <~~ reading 4444 of SRL is like reading 8888 of RL    |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       . x4444 .       .       .       .       .       |
                                                                #   &<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
    #     017 [0x4444 : RL = SB[6] & SRL;]                      # x4444 .       .       .       .       .       .       .       |
    RL["0x4444"] <= sb6() & SRL()                               # 6 & S .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
                                                                # x4444 . <~~ reading 8888 of NRL is like reading 4444 of RL    |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       . x8888 .       .       .       .       .       |
                                                                #  OR<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
    #     018 [0x4444 : SB[3] = RL;;0x8888 : RL = SB[6] | NRL;] #   v   .       .       .       .       .       .       .       |
    with apl_commands("compatible sections: 8th instance"):     # x8888 .       .       .       .       .       .       .       |
        RL ["0x8888"] <= sb6() | NRL()                          # 6 | N .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x8888 .       .       .       .       .       .       .       |
                                                                #   o>------------->,   . <~~ write     .       .       .       |
                                                                #       .       .   |   .       .       .       .       .       |
                                                                #       . x4444 .   |   .       .       .       .       .       |
                                                                #   ,<-----<o   .   |   . <~~ read      .       .       .       |
    #     019 [0x8888 : SB[6] = RL;;0x4444 : RL = SB[5];]       #   |   .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 9th instance"):     #   v   .       .   v   .       .       .       .       .       |
        sb6["0x8888"] <= RL()                                   # x4444 .       . x8888 .       .       .       .       .       |
        RL ["0x4444"] <= sb5()                                  #   5   .       .   R   .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x4444 .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   . x8888 .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
    #     020 [0x8888 : RL = SB[5] & NRL;]                      # x8888 .       .       .       .       .       .       .       |
    RL["0x8888"] <= sb5() & NRL()                               # 5 & N .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x8888 .       .       .       .       .       .       .       |
                                                                #   o>----->,   . <~~ write     .       .       .       .       |
                                                                #       .   |   .       .       .       .       .       .       |
                                                                #  "0"  .   |   . <~~ read      .       .       .       .       |
    #    021 [0x8888 : SB[3,5] = RL;]                           #   |   .   |   .       .       .       .       .       .       |
    with apl_commands("read after write; special instance"):    #   v   .   v   .       .       .       .       .       .       |
        sb5["0x8888"] <= RL()                                   # x000f . x8888 .       .       .       .       .       .       |
        RL["0x000F"] <= 0                                       #  "0"  .   R   .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
                                                                # x000f .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       . x000f .       .       .       .       .       |
                                                                #  OR<-------------<o   . <~~ read      .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
                                                                # x0008 .       . ~~> broadcast .       .       .       .       |
                                                                #   o>--------------------------------------------->,   .       |
    #     023 [0x000f : RL |= SB[6];]                           #   |   .       .       .       .       .       .   |   .       |
    with apl_commands("broadcast after read: 1st instance"):    #   v   .       .       .       .       .       .   v   .       |
        RL["0x000F"] |= sb6()                                   # x000f .       .       .       .       .       . x0008 .       |
        GL["0x0008"] <= RL()  # new RL, RL*                     # 6 | R .       .       .       .       .       .   R*  .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x0007 . <~~ reading 000e of NRL is like reading 0007 of RL    |
                                                                #   o>------------->,   .       .       .       .       .       |
                                                                #       .       .   |   .       .       .       .       .       |
                                                                #  "0"  . <~~ read  |   .       .       .       .       .       |
    #     027 [0x000e : SB[6] = NRL;;0x0001 : RL = SB[7];]      #   |   .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 10th instance"):    #   v   .       .   v   .       .       .       .       .       |
        sb6["0x000e"] <= NRL()                                  # x0001 .       . x000e . <~~ write     .       .       .       |
        RL ["0x0001"] <= 0                                      #   7   .       .   N   .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
    #     028 [0x0001 : SB[6] = RL;;0xffff : RL = SB[4];]       # x0001 .       .       .       .       .       .       .       |
    with apl_commands():                                        #   o>----------> x0001 .       .       .       .       .       |
        sb6["0x0001"] <= RL()                                   #       .       .   R   .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       . x00f0 .       |
                                                                #   ,<---------------------------------------------<o   .       |
                                                                #   v   . x00f0 .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       |
    #     030 [0x00f0 : RL = SB[5,7];]                          #   |   .       .       .       .       .       .       .       |
    with apl_commands("compatible sections; 11th instance?"):   #   v   .       .       .       .       .       .       .       |
        RL["0x00f0"] <= sb5() & GL()  # sb7()                   # x00f0 .       .       .       .       .       .       .       |
        sb7[:]       <= GL()                                    # 5 & G .       .       .   G <--------------------<o   .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
                                                                # x00f0 .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       . x00f0 .       .       .       .       .       |
                                                                #  OR<-------------<o   . <~~ read      .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
                                                                # x0080 .       . ~~> broadcast .       .       .       .       |
                                                                #   o>--------------------------------------------->,   .       |
    #     031 [0x00f0 : RL |= SB[6];]                           #   |   .       .       .       .       .       .   |   .       |
    with apl_commands("broadcast after read"):                  #   v   .       .       .       .       .       .   v   .       |
        RL["0x00f0"] |= sb6()                                   # x00f0 .       .       .       .       .       . x0080 .       |
        GL["0x0080"] <= RL()                                    # 6 | R .       .       .       .       .       .   R*  .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x0070 . <~~ write     .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       |
                                                                #       .       .   |   . x0010 .       .       .       .       |
                                                                #   ,<--------------|------<o   . <~~ read      .       .       |
    #     035 [0x00e0 : SB[6] = NRL;;0x0010 : RL = SB[7];]      #   |   .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 11th instance"):    #   v   .       .   v   .       .       .       .       .       |
        sb6["0x00e0"] <= NRL()                                  # x0010 .       . x00e0 .       .       .       .       .       |
        RL ["0x0010"] <= sb7()                                  #   7   .       .   N   .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
    #     036 [0x0010 : SB[6] = RL;;0xffff : RL = SB[4];]       # x0010 .       .       .       .       .       .       .       |
    with apl_commands():                                        #   o>----------> x0100 .       .       .       .       .       |
        sb6["0x0010"] <= RL()                                   #       .       .   R   .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       . x0f00 .       |
                                                                #   ,<---------------------------------------------<o   .       |
                                                                #   |   . x0f00 .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       |
    #     038 [0x0f00 : RL = SB[5,7];]  # OPPORTUNITY?          #   |   .       .       .       .       .       .       .       |
    with apl_commands():                                        #   v   .       .       .       .       .       .       .       |
        RL["0x0f00"] <= sb5() & GL()  # sb7()                   # x0f00 .       .       .       .       .       .       .       |
        sb7[:]       <= GL()  # RL()                            # 5 & G .       .       .   G <--------------------<o   .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
                                                                # x0f00 .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       . x0f00 .       .       .       .       .       |
                                                                #  OR<-------------<o   . <~~ read      .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
                                                                # x0800 .       . ~~> broadcast .       .       .       .       |
                                                                #   o>--------------------------------------------->,00 .       |
    #     039 [0x0f00 : RL |= SB[6];]                           #   |   .       .       .       .       .       .   |   .       |
    with apl_commands():                                        #   v   .       .       .       .       .       .   v   .       |
        RL["0x0f00"] |= sb6()                                   # x00f0 .       .       .       .       .       . x0800 .       |
        GL["0x0800"] <= RL()                                    # 6 | R .       .       .               .       .   R   .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x0700 . <~~ write     .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       |
                                                                #       .       .   |   . x0100 .       .       .       .       |
                                                                #   ,<--------------|------<o   . <~~ read      .       .       |
    #     043 [0x0e00 : SB[6] = NRL;;0x0100 : RL = SB[7];]      #   |   .       .   |   .       .       .       .       .       |
    with apl_commands("compatible sections: 12th instance"):    #   v   .       .   v   .       .       .       .       .       |
        sb6["0x0e00"] <= NRL()                                  # x0100 .       . x0e00 .       .       .       .       .       |
        RL ["0x0100"] <= sb7()                                  #   7   .       .   N   .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
    #     044 [0x0100 : SB[6] = RL;;0xffff : RL = SB[4];]       # x0100 .       .       .       .       .       .       .       |
    with apl_commands("read after write: 5th instance"):        #   o>----------> x0100 .       .       .       .       .       |
        sb6["0x0100"] <= RL()                                   #       .       .   R   .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       . xf000 .       |
                                                                #   ,<---------------------------------------------<o   .       |
                                                                #   |   . xf000 .       .       .       .       .       .       |
                                                                #   &<-----<o   .       .       .       .       .       .       |
    #     046 [0xf000 : RL = SB[5,7];]                          #   |   .       .       .       .       .       .       .       |
    # UNREALISTIC TO REPLACE THE ALIAS OF GL FOR SB7            #   v   .       .       .       .       .       .       .       |
    with apl_commands():                                        # xf000 .       .       .       .       .       .       .       |
        RL["0xf000"] <= sb5() & GL()  # sb7()                   # 5 & G .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #   R   |  SB5  |  SB6  |  SB7  |  SB1  |  SB2  |  GL   |  out  |
                                                                #-------|-------|-------|-------|-------|-------|-------|-------|
                                                                #       .       .       .       .       .       .       .       |
                                                                # xf000 .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       . xf000 .       .       .       .       .       |
                                                                #  OR<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                #   v   .       .       .       .       .       .       .       |
    #     047 [0xf000 : RL |= SB[6];]                           # xf000 .       .       .       .       .       .       .       |
    RL["0xf000"] |= sb6()                                       # 6 | R .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                # x7000 .       .       .       .       .       .       .       |
                                                                #   o>------------->,   .       .       .       .       .       |
                                                                #       .       .   |   .       .       .       .       .       |
                                                                #       .       .   v   .       .       .       .       .       |
                                                                #       .       . xe000 .       .       .       .       .       |
    #     052 [0x1000 : SB[6] = RL;;0xffff : RL = SB[1];]       #       .       .   |   .       .       .       .       .       |
    with apl_commands("read after write: 5th instance"):        #   ,<--------------|--------------<o   .       .       .       |
        sb6["0x1000"] <= GL()  # RL()                           #   |   .       .   v   .       .       .       . x1000 .       |
        RL [   ::   ] <= sb1()                                  #   v   .       . x1000 <--------------------------<o   .       |
        sb6["0xe000"] <= NRL()                                  #   1   .       .   G   .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                # XOR<-------------<o   .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
    #     053 [0xffff : RL ^= SB[6];]                           #   v   .       .       .       .       .       .       .       |
    RL[::] ^= sb6()                                             # 6 ^ R .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
                                                                #   o   .       .       .       .       .       .       .       |
                                                                #   |   .       .       .       .       .       .       .       |
                                                                # XOR<-------------------------------------<o   .       .       |
    #     054 [0xffff : SB[3] = RL;;0xffff : RL = SB[2];]       #   |   .       .       .       .       .       .       .       |
    with apl_commands("read after write: 6th instance"):        #   v   .       .       .       .       .       .       .       |
        RL [:] ^= sb2()                                         # 2 ^ R .       .       .       .       .       .       .       |
                                                                #       .       .       .       .       .       .       .       |
    out[:] <= RL()                                              #   o>----------------------------------------------------> R   |
                                                                #       .       .       .       .       .       .       .       |
                                                                #---------------------------------------------------------------|


@belex_apl
def laning_laboratory_011(Belex, out: VR, sb1: VR, sb2: VR):
    sb4 = RN_REG_T4  # resurrect this old dead reg
    sb3 = RN_REG_T3  # resurrect
    sb5 = RN_REG_T5
    sb6 = RN_REG_T6
    sb7 = RN_REG_T0

    RL[:] <= 0
    #     003 [0xffff : SB[7] = RL;;0xffff : RL = SB[1,2];]
    with apl_commands("read after write, 1st instance"):
        sb7[:] <= RL()
        RL [:] <= sb1() & sb2()
    #     004 [0xffff : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write, 2nd instance"):
        sb6[:] <= RL()
        RL [:] <= sb1()
    #     005 [0xffff : RL ^= SB[2];]
    RL[:] ^= sb2()
    #     006 [0xffff : SB[3,5] = RL;]
    sb5[:] <= RL()
    sb4[:] <= RL()

    # RL[:] <= 0
    # RL["0x1111"] <= sb5()
    # RL["0x2222"] &= NRL()
    # RL["0x4444"] &= NRL()
    # RL["0x8888"] &= NRL()
    # sb5[:] <= RL()

    # sb5 begins as x^y; sb6 begins as x&y

    # Three commands, four lines setting up sb6
    # RL ["0x1111"] <= sb6() & SRL()  # RL[1] := sb6[1] & sb5[2]
    # RL ["0x2222"] <= sb6() | NRL()  # RL[2] := sb6[2] | (sb6[1] & sb5[2])
    # with apl_commands():
    #     sb6["0x2222"] <= RL()       # sb6[2] := sb6[2] | (sb6[1] & sb5[2])
    # # Four commands, five lines setting up sb5
    #     RL ["0x1111"] <= sb5()
    # RL ["0x2222"] <= sb5() & NRL()
    # with apl_commands():
    #     sb5["0x2222"] <= RL()  # really sb6 in disguise
    #     RL ["0x4444"] <= sb5()
    # # Three commands, four lines setting up sb6
    # RL ["0x2222"] <= sb6() & SRL()
    # RL ["0x4444"] <= sb6() | NRL()
    # with apl_commands():
    #     sb6["0x4444"] <= RL()
    #     # Four commands, five lines setting up sb5
    #     RL ["0x2222"] <= sb5()
    # RL ["0x4444"] <= sb5() & NRL()
    # with apl_commands():
    #     sb5["0x4444"] <= RL()  # really sb6 in disguise
    #     RL ["0x8888"] <= sb5()
    # # Three commands, four lines setting up sb6
    # RL ["0x4444"] <= sb6() & SRL()
    # RL ["0x8888"] <= sb6() | NRL()
    # with apl_commands():
    #     sb6["0x8888"] <= RL()
    #     # Three commands, three lines setting up sb5
    #     RL ["0x4444"] <= sb5()
    # RL ["0x8888"] <= sb5() & NRL()
    # sb5["0x8888"] <= RL()  # really sb6 in disguise
    # # xxxxxxxx

    # RL == sb5 = x^y; sb6 = x&y
    # RL ["0x1111"] <= sb6() & SRL()  # RL[1]  := sb6[1] & sb5[2]
    # RL ["0x2222"] <= sb6() | NRL()  # RL[2]  := sb6[2] | (sb6[1] & sb5[2])
    # sb6["0x2222"] <= RL()           # sb6[2] := sb6[2] | (sb6[1] & sb5[2])
    #
    # RL ["0x1111"] <= sb5()
    # RL ["0x2222"] <= sb5() & NRL()
    # sb5["0x2222"] <= RL()
    # RL ["0x4444"] <= sb5()
    #
    # RL ["0x2222"] <= sb6() & SRL()  # RL[2]  := sb6[2] & sb5[4]
    # RL ["0x4444"] <= sb6() | NRL()  # RL[4]  := sb6[4] & (sb6[2] & sb5[4])
    # sb6["0x4444"] <= RL()           # sb6[4] := sb6[4] | (sb6[2] & sb5[4])
    #
    # RL ["0x2222"] <= sb5()
    # RL ["0x4444"] <= sb5() & NRL()
    # sb5["0x4444"] <= RL()
    # RL ["0x8888"] <= sb5()
    #
    # RL ["0x4444"] <= sb6() & SRL()  # RL[4]  := sb6[4] & sb5[8]
    # RL ["0x8888"] <= sb6() | NRL()  # RL[8]  := sb6[8] & (sb6[4] & sb5[8])
    # sb6["0x8888"] <= RL()           # sb6[8] := sb6[8] & (sb6[4] & sb5[8])
    #
    # RL ["0x4444"] <= sb5()
    # RL ["0x8888"] <= sb5() & NRL()
    # sb5["0x8888"] <= RL()
    # xxxxxxxx

    # RL == sb5 = x^y; sb6 = x&y

    ## RL ["0x1111"] <= sb5()          # RL [1] := sb5[1]
    ## RL ["0x2222"] <= sb5() & NRL()  # RL [2] := sb5[2] & sb5[1]
    ## sb5["0x2222"] <= RL()           # sb5[2] := sb5[2] & sb5[1]

    ## RL ["0x2222"] <= sb5()          # RL [2] := sb5[2]
    ## RL ["0x4444"] <= sb5() & NRL()  # RL [4] := sb5[4] & sb5[2]
    ## sb5["0x4444"] <= RL()           # sb5[4] := sb5[4] & sb5[2]

    ## RL ["0x4444"] <= sb5()          # RL [4] := sb5[4]
    ## RL ["0x8888"] <= sb5() & NRL()  # RL [8] := sb5[8] & sb5[4]
    ## sb5["0x8888"] <= RL()           # sb5[8] := sb5[8] & sb5[4]

    # RL ["0x1111"] <= sb5()          # RL [1] := sb5[1]
    # RL ["0x2222"] <= sb5() & NRL()  # RL [2] := sb5[2] & sb5[1]
    # sb5["0x2222"] <= RL()           # sb5[2] := sb5[2] & sb5[1]

    # RL ["0x2222"] <= sb5()          # RL [2] := sb5[2]
    # RL ["0x4444"] <= sb5() & NRL()  # RL [4] := sb5[4] & sb5[2]
    # sb5["0x4444"] <= RL()           # sb5[4] := sb5[4] & sb5[2]

    # RL ["0x4444"] <= sb5()          # RL [4] := sb5[4]
    # RL ["0x8888"] <= sb5() & NRL()  # RL [8] := sb5[8] & sb5[4]
    # sb5["0x8888"] <= RL()           # sb5[8] := sb5[8] & sb5[4]

    # RL ["0xFFFF"] <= sb5()
    with apl_commands():
        RL ["0xAAAA"] &= NRL()
        GGL["0x2222"] <= RL()
    RL ["0xCCCC"] &= GGL()

    sb5["0xFFFF"] <= RL()

    # RL [:] <= sb4()
    # sb3[:] <= NRL()

    RL [:] <= sb6()
    RL ["0x2222"] |= sb4() & NRL()
    RL ["0x4444"] |= sb4() & NRL()
    RL ["0x8888"] |= sb4() & NRL()
    sb6[:] <= RL()

    # for i == 2, 4, 8
    #   RL[i]  := sb6[i] | (sb6[i-1] & sb4[i])  # Leo says
    #   sb6[i] := RL[i]
    #

    # RL [:] <= sb6()
    # sb6[:] <= ~RL()
    # RL [:] <= sb4()
    # sb4[:] <= ~RL()

    # with apl_commands():
    #     RL ["0x3333"] <= sb6()
    #     GGL["0x3333"] <= RL()
    # RL ["0x2222"] <= (sb4() & sb6()) | GGL()
    # sb6["0x2222"] <= RL()

    # with apl_commands():
    #     RL ["0x6666"] <= sb6()
    #     GGL["0x6666"] <= RL()
    # RL ["0x4444"] <= (sb4() & sb6()) | GGL()
    # sb6["0x4444"] <= RL()

    # with apl_commands():
    #     RL ["0xCCCC"] <= sb6()
    #     GGL["0xCCCC"] <= RL()
    # RL ["0x8888"] <= (sb4() & sb6()) | GGL()
    # sb6["0x8888"] <= RL()

    # RL [:] <= sb6()
    # sb6[:] <= ~RL()
    # RL [:] <= sb4()
    # sb4[:] <= ~RL()

    # RL ["0x2222"] <= sb4()          # RL [2] := x^y[2]

    # RL ["0x1111"] <= sb6() & SRL()  # RL [1] := sb6[1] & x^y[2]
    # RL ["0x2222"] <= sb6() | NRL()  # RL [2] := sb6[2] | (sb6[1] & x^y[2])
    # sb6["0x2222"] <= RL()           # sb6[2] := sb6[2] | (sb6[1] & x^y[2])

    # RL ["0x4444"] <= sb4()          # RL [4] := x^y[4]

    # RL ["0x2222"] <= sb6() & SRL()  # RL [2] := sb6[2] & x^y[4]
    # RL ["0x4444"] <= sb6() | NRL()  # RL [4] := sb6[4] & (sb6[2] & x^y[4])
    # sb6["0x4444"] <= RL()           # sb6[4] := sb6[4] | (sb6[2] & x^y[4])

    # RL ["0x8888"] <= sb4()          # RL [8] := x^y[8]

    # RL ["0x4444"] <= sb6() & SRL()  # RL [4] := sb6[4] & x^y[8]
    # RL ["0x8888"] <= sb6() | NRL()  # RL [8] := sb6[8] & (sb6[4] & x^y[8])
    # sb6["0x8888"] <= RL()           # sb6[8] := sb6[8] & (sb6[4] & x^y[8])


    ## with apl_commands():
    ##     RL ["0x1111"] <= sb5()          # RL [1] := sb5[1]
    ##     RL ["0x2222"] <= sb5() & NRL()  # RL [2] := sb5[2] & sb5[1]
    ## sb5["0x2222"] <= RL()           # sb5[2] := sb5[2] & sb5[1]

    ## with apl_commands():
    ##     RL ["0x2222"] <= sb5()          # RL [2] := sb5[2]
    ##     RL ["0x4444"] <= sb5() & NRL()  # RL [4] := sb5[4] & sb5[2]
    ## sb5["0x4444"] <= RL()           # sb5[4] := sb5[4] & sb5[2]

    ## with apl_commands():
    ##     RL ["0x4444"] <= sb5()          # RL [4] := sb5[4]
    ##     RL ["0x8888"] <= sb5() & NRL()  # RL [8] := sb5[8] & sb5[4]
    ## with apl_commands():
    ##     sb5["0x8888"] <= RL()           # sb5[8] := sb5[8] & sb5[4]

    ##     RL ["0x2222"] <= sb4()
    ## RL ["0x1111"] <= sb6() & SRL()  # RL [1] := sb6[1] & x^y[2]
    ## RL ["0x2222"] <= sb6() | NRL()  # RL [2] := sb6[2] | (sb6[1] & x^y[2])
    ## with apl_commands():
    ##     sb6["0x2222"] <= RL()           # sb6[2] := sb6[2] | (sb6[1] & x^y[2])

    ##     RL ["0x4444"] <= sb4()          # RL [4] := x^y[4]
    ## RL ["0x2222"] <= sb6() & SRL()  # RL [2] := sb6[2] & x^y[4]
    ## RL ["0x4444"] <= sb6() | NRL()  # RL [4] := sb6[4] & (sb6[2] & x^y[4])
    ## with apl_commands():
    ##     sb6["0x4444"] <= RL()           # sb6[4] := sb6[4] | (sb6[2] & x^y[4])

    ##     RL ["0x8888"] <= sb4()          # RL [8] := x^y[8]
    ## RL ["0x4444"] <= sb6() & SRL()  # RL [4] := sb6[4] & x^y[8]
    ## RL ["0x8888"] <= sb6() | NRL()  # RL [8] := sb6[8] & (sb6[4] & x^y[8])
    ## sb6["0x8888"] <= RL()           # sb6[8] := sb6[8] & (sb6[4] & x^y[8])

    # xxxxxxxx

    #     007 [0x2222 : RL = SB[5];;0x1111 : RL = SB[6] & SRL;]
    # with apl_commands("compatible sections: 1st instance"):
    #     RL["0x1111"] <= sb6() & SRL()  # uses old RL
    # #     008 [0x1111 : SB[3] = RL;;0x2222 : RL = SB[6] | NRL;]
    # with apl_commands("compatible sections: 2nd instance"):
    #     RL ["0x2222"] <= sb6() | NRL()  # uses old RL, th'fore
    #     009 [0x2222 : SB[6] = RL;;0x1111 : RL = SB[5];]
    # with apl_commands("compatible sections: 3rd instance"):
    #     # sb6["0x2222"] <= RL()
    #     RL ["0x1111"] <= sb5()
    # #     010 [0x2222 : RL = SB[5] & NRL;]
    # RL["0x2222"] <= sb5() & NRL()
    # #     011 [0x2222 : SB[5] = RL;;0x4444 : RL = SB[5];]
    # with apl_commands("compatible sections: 4th instance"):
    #     sb5["0x2222"] <= RL()
    #     RL ["0x4444"] <= sb5()
    # #    012 [0x2222 : RL = SB[6] & SRL;]
    # RL["0x2222"] <= sb6() & SRL()
    # # can't be moved up "attempt to update wr controls"
    # #     013 [0x2222 : SB[3] = RL;;0x4444 : RL = SB[6] | NRL;]
    # with apl_commands("compatible sections: 5th instance"):
    #     RL ["0x4444"] <= sb6() | NRL()  # Accesses RL["0x2222"] from Instr 12.
    # #     014 [0x4444 : SB[6] = RL;;0x2222 : RL = SB[5];]
    # with apl_commands("compatible sections: 6th instance"):
    #     # sb6["0x4444"] <= RL()
    #     RL ["0x2222"] <= sb5()
    # #     015 [0x4444 : RL = SB[5] & NRL;]
    # RL["0x4444"] <= sb5() & NRL()
    # #     016 [0x4444 : SB[5] = RL;;0x8888 : RL = SB[5];]
    # with apl_commands("compatible sections: 7th instance"):
    #     sb5["0x4444"] <= RL()
    #     RL ["0x8888"] <= sb5()
    # #     017 [0x4444 : RL = SB[6] & SRL;]
    # RL["0x4444"] <= sb6() & SRL()
    # #     018 [0x4444 : SB[3] = RL;;0x8888 : RL = SB[6] | NRL;]
    # with apl_commands("compatible sections: 8th instance"):
    #     RL ["0x8888"] <= sb6() | NRL()
    # #     019 [0x8888 : SB[6] = RL;;0x4444 : RL = SB[5];]
    # with apl_commands("compatible sections: 9th instance"):
    #     sb6["0x8888"] <= RL()
    #     RL ["0x4444"] <= sb5()
    #     020 [0x8888 : RL = SB[5] & NRL;]
    # RL["0x8888"] <= sb5() & NRL()
    #    021 [0x8888 : SB[3,5] = RL;]
    with apl_commands("read after write; special instance"):
        # sb5["0x8888"] <= RL()
        RL["0x000F"] <= 0
    #     023 [0x000f : RL |= SB[6];]
    with apl_commands("broadcast after read: 1st instance"):
        RL["0x000F"] |= sb6()
        GL["0x0008"] <= RL()  # new RL, RL*
    #     027 [0x000e : SB[6] = NRL;;0x0001 : RL = SB[7];]
    with apl_commands("compatible sections: 10th instance"):
        sb6["0x000e"] <= NRL()
        RL ["0x0001"] <= 0
    #     028 [0x0001 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands():
        sb6["0x0001"] <= RL()
    #     030 [0x00f0 : RL = SB[5,7];]
    with apl_commands("compatible sections; 11th instance?"):
        RL["0x00f0"] <= sb5() & GL()  # sb7()
        sb7[:]       <= GL()
    #     031 [0x00f0 : RL |= SB[6];]
    with apl_commands("broadcast after read"):
        RL["0x00f0"] |= sb6()
        GL["0x0080"] <= RL()
    #     035 [0x00e0 : SB[6] = NRL;;0x0010 : RL = SB[7];]
    with apl_commands("compatible sections: 11th instance"):
        sb6["0x00e0"] <= NRL()
        RL ["0x0010"] <= sb7()
    #     036 [0x0010 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands():
        sb6["0x0010"] <= RL()
    #     038 [0x0f00 : RL = SB[5,7];]  # OPPORTUNITY?
    with apl_commands():
        RL["0x0f00"] <= sb5() & GL()  # sb7()
        sb7[:]       <= GL()  # RL()
    #     039 [0x0f00 : RL |= SB[6];]
    with apl_commands():
        RL["0x0f00"] |= sb6()
        GL["0x0800"] <= RL()
    #     043 [0x0e00 : SB[6] = NRL;;0x0100 : RL = SB[7];]
    with apl_commands("compatible sections: 12th instance"):
        sb6["0x0e00"] <= NRL()
        RL ["0x0100"] <= sb7()
    #     044 [0x0100 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x0100"] <= RL()
    #     046 [0xf000 : RL = SB[5,7];]
    # UNREALISTIC TO REPLACE THE ALIAS OF GL FOR SB7
    with apl_commands():
        RL["0xf000"] <= sb5() & GL()  # sb7()
    #     047 [0xf000 : RL |= SB[6];]
    RL["0xf000"] |= sb6()
    #     052 [0x1000 : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x1000"] <= GL()  # RL()
        RL [   ::   ] <= sb1()
        sb6["0xe000"] <= NRL()
    #     053 [0xffff : RL ^= SB[6];]
    RL[::] ^= sb6()
    #     054 [0xffff : SB[3] = RL;;0xffff : RL = SB[2];]
    with apl_commands("read after write: 6th instance"):
        RL [:] ^= sb2()

    out[:] <= RL()



@belex_apl
def laning_laboratory_001(Belex, out: VR, sb1: VR, sb2: VR):
    sb4 = RN_REG_T4  # resurrect this old dead reg
    sb3 = RN_REG_T3  # resurrect
    sb5 = RN_REG_T5
    sb6 = RN_REG_T6
    sb7 = RN_REG_T0

    RL[:] <= 0

    #     003 [0xffff : SB[7] = RL;;0xffff : RL = SB[1,2];]
    with apl_commands("read after write, 1st instance"):
        sb7[:] <= RL()
        RL [:] <= sb1() & sb2()

    #     004 [0xffff : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write, 2nd instance"):
        sb6[:] <= RL()
        RL [:] <= sb1()

    #     005 [0xffff : RL ^= SB[2];]
    RL[:] ^= sb2()

    #     006 [0xffff : SB[3,5] = RL;]
    with apl_commands():
        sb5[:] <= RL()
        sb4[:] <= RL()

    # RL ["0xFFFF"] <= sb5()
    with apl_commands():
        RL ["0xAAAA"] &= NRL()
        GGL["0x2222"] <= RL()
    RL ["0xCCCC"] &= GGL()

    with apl_commands("read after write; special instance 1"):
        sb5["0xFFFF"] <= RL()
        RL [:] <= sb6()

    RL ["0x2222"] |= sb4() & NRL()
    RL ["0x4444"] |= sb4() & NRL()
    RL ["0x8888"] |= sb4() & NRL()

    #    021 [0x8888 : SB[3,5] = RL;]
    with apl_commands("read after write; special instance 2"):
        # sb5["0x8888"] <= RL()
        sb6[:] <= RL()
        RL["0x000F"] <= 0

    #     023 [0x000f : RL |= SB[6];]
    with apl_commands("broadcast after read: 1st instance"):
        RL["0x000F"] |= sb6()
        GL["0x0008"] <= RL()  # new RL, RL*

    #     027 [0x000e : SB[6] = NRL;;0x0001 : RL = SB[7];]
    with apl_commands("compatible sections: 10th instance"):
        sb6["0x000e"] <= NRL()
        RL ["0x0001"] <= 0

    #     028 [0x0001 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands():
        sb6["0x0001"] <= RL()

    #     030 [0x00f0 : RL = SB[5,7];]
    with apl_commands("compatible sections; 11th instance?"):
        RL["0x00f0"] <= sb5() & GL()  # sb7()
        sb7[:]       <= GL()

    #     031 [0x00f0 : RL |= SB[6];]
    with apl_commands("broadcast after read"):
        RL["0x00f0"] |= sb6()
        GL["0x0080"] <= RL()

    #     035 [0x00e0 : SB[6] = NRL;;0x0010 : RL = SB[7];]
    with apl_commands("compatible sections: 11th instance"):
        sb6["0x00e0"] <= NRL()
        RL ["0x0010"] <= sb7()

    #     036 [0x0010 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands():
        sb6["0x0010"] <= RL()

    #     038 [0x0f00 : RL = SB[5,7];]  # OPPORTUNITY?
    with apl_commands():
        RL["0x0f00"] <= sb5() & GL()  # sb7()
        sb7[:]       <= GL()  # RL()

    #     039 [0x0f00 : RL |= SB[6];]
    with apl_commands():
        RL["0x0f00"] |= sb6()
        GL["0x0800"] <= RL()

    #     043 [0x0e00 : SB[6] = NRL;;0x0100 : RL = SB[7];]
    with apl_commands("compatible sections: 12th instance"):
        sb6["0x0e00"] <= NRL()
        RL ["0x0100"] <= sb7()

    #     044 [0x0100 : SB[6] = RL;;0xffff : RL = SB[4];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x0100"] <= RL()

    #     046 [0xf000 : RL = SB[5,7];]
    # UNREALISTIC TO REPLACE THE ALIAS OF GL FOR SB7
    with apl_commands():
        RL["0xf000"] <= sb5() & GL()  # sb7()

    #     047 [0xf000 : RL |= SB[6];]
    RL["0xf000"] |= sb6()

    #     052 [0x1000 : SB[6] = RL;;0xffff : RL = SB[1];]
    with apl_commands("read after write: 5th instance"):
        sb6["0x1000"] <= GL()  # RL()
        RL [   ::   ] <= sb1()
        sb6["0xe000"] <= NRL()

    #     053 [0xffff : RL ^= SB[6];]
    RL[::] ^= sb6()

    #     054 [0xffff : SB[3] = RL;;0xffff : RL = SB[2];]
    with apl_commands("read after write: 6th instance"):
        RL [:] ^= sb2()

    out[:] <= RL()



@belex_property_test(laning_laboratory_001)
def test_laning_laboratory_001(sb1: np.ndarray, sb2: np.ndarray) -> np.ndarray:
    r"""Test HLB adder (Dima's) with random data."""
    return sb1 + sb2


# 001 [0xffff : RL = SB[1];]
# 002 [0xffff : RL ^= SB[1];]
# 003 [0xffff : SB[7] = RL;;0xffff : RL = SB[1,2];]
# 004 [0xffff : SB[6] = RL;;0xffff : RL = SB[1];]
# 005 [0xffff : RL ^= SB[2];]
# 006 [0xffff : SB[3,5] = RL;]
# 007 [0x2222 : RL = SB[5];;0x1111 : RL = SB[6] & SRL;]
# 008 [0x1111 : SB[3] = RL;;0x2222 : RL = SB[6] | NRL;]
# 009 [0x2222 : SB[6] = RL;;0x1111 : RL = SB[5];]
# 010 [0x2222 : RL = SB[5] & NRL;]
# 011 [0x2222 : SB[5] = RL;;0x4444 : RL = SB[5];]
# 012 [0x2222 : RL = SB[6] & SRL;]
# 013 [0x2222 : SB[3] = RL;;0x4444 : RL = SB[6] | NRL;]
# 014 [0x4444 : SB[6] = RL;;0x2222 : RL = SB[5];]
# 015 [0x4444 : RL = SB[5] & NRL;]
# 016 [0x4444 : SB[5] = RL;;0x8888 : RL = SB[5];]
# 017 [0x4444 : RL = SB[6] & SRL;]
# 018 [0x4444 : SB[3] = RL;;0x8888 : RL = SB[6] | NRL;]
# 019 [0x8888 : SB[6] = RL;;0x4444 : RL = SB[5];]
# 020 [0x8888 : RL = SB[5] & NRL;]
# 021 [0x8888 : SB[3,5] = RL;]
# 022 [0x000f : RL = SB[5,7];]
# 023 [0x000f : RL |= SB[6];]
# 024 [0x000f : SB[3,6] = RL;]
# 025 [0x0008 : RL = SB[6];;0x0008 : GL = RL;]
# 026 [0xffff : SB[4] = GL;;0x0007 : RL = SB[6];]
# 027 [0x000e : SB[6] = NRL;;0x0001 : RL = SB[7];]
# 028 [0x0001 : SB[6] = RL;;0xffff : RL = SB[4];]
# 029 [0xffff : SB[7] = RL;]
# 030 [0x00f0 : RL = SB[5,7];]
# 031 [0x00f0 : RL |= SB[6];]
# 032 [0x00f0 : SB[3,6] = RL;]
# 033 [0x0080 : RL = SB[6];;0x0080 : GL = RL;]
# 034 [0xffff : SB[4] = GL;;0x0070 : RL = SB[6];]
# 035 [0x00e0 : SB[6] = NRL;;0x0010 : RL = SB[7];]
# 036 [0x0010 : SB[6] = RL;;0xffff : RL = SB[4];]
# 037 [0xffff : SB[7] = RL;]
# 038 [0x0f00 : RL = SB[5,7];]
# 039 [0x0f00 : RL |= SB[6];]
# 040 [0x0f00 : SB[3,6] = RL;]
# 041 [0x0800 : RL = SB[6];;0x0800 : GL = RL;]
# 042 [0xffff : SB[4] = GL;;0x0700 : RL = SB[6];]
# 043 [0x0e00 : SB[6] = NRL;;0x0100 : RL = SB[7];]
# 044 [0x0100 : SB[6] = RL;;0xffff : RL = SB[4];]
# 045 [0xffff : SB[7] = RL;]
# 046 [0xf000 : RL = SB[5,7];]
# 047 [0xf000 : RL |= SB[6];]
# 048 [0xf000 : SB[6] = RL;]
# 049 [0x8000 : RL = SB[6];;0x8000 : GL = RL;]
# 050 [0xffff : SB[4] = GL;;0x7000 : RL = SB[6];]
# 051 [0xe000 : SB[6] = NRL;;0x1000 : RL = SB[7];]
# 052 [0x1000 : SB[6] = RL;;0xffff : RL = SB[1];]
# 053 [0xffff : RL ^= SB[6];]
# 054 [0xffff : SB[3] = RL;;0xffff : RL = SB[2];]
# 055 [0xffff : RL ^= SB[3];]
# 056 [0xffff : SB[0,3] = RL;;0xffff : RL = SB[4];]
# 057 [0xffff : SB[7] = RL;]
# OrderedDict([('IR', 8), ('out', 0), ('x', 2), ('y', 1), ('_INTERNAL0', 6), ('_INTERNAL1', 5), ('_INTERNAL2', 7), ('_INTERNAL3', 4), ('t_0', 3), ('t_1', 3), ('t_12', 3), ('t_13', 3), ('t_2', 3), ('t_20', 3), ('t_21', 3), ('t_3', 3), ('t_5', 3), ('t_6', 3)])


# hot fix for Eyal

@belex_apl
def process(Belex, src0: VR, src1: VR, src2: VR) -> None:
    global SM_0XFFFFF
    cpy_imm_16(src1, "012")  # "0x7")
    cpy_imm_16(src2, "8888")  # "0x10")
    or_16(src0, src1, src2)
    rl_from_sb(src0)
    rsp_out(SM_0XFFFF)
