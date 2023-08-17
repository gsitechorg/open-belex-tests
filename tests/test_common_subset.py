r"""
By Brian Beckman
"""

import pytest

from hypothesis import given, settings

from open_belex.common.constants import NUM_PLATS_PER_APUC, NUM_HALF_BANKS_PER_APUC
from open_belex.common.mask import Mask
from open_belex.common.subset import *
from open_belex_tests.strategies import *


# Set the max_examples to be a larger number to generate more
# tests. Set it low to run the tests faster.
g_max_examples = 10


NIBBLE_COUNT_TABLE = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]


def count_bits(hex: str):
    result = reduce(
        lambda c, x: c + NIBBLE_COUNT_TABLE[int(x, base=16)],
        hex, 0)
    return result


def assert_for_a_mex(mex: int, user_input):

    victim0, victim1 = assert_via_binary_strings(mex, user_input)

    assert_as_full_integers(victim0, victim1)

    assert_as_hex(victim0, victim1)

    assert_inverted(victim0, victim1)

    assert_shifted(victim0, victim1)

    assert_from_inv_hex(mex, victim0)

    assert_from_full_integer(mex, victim0)

    assert_bit_count(victim0)

    debug_me = 42


def assert_via_binary_strings(mex, user_input) -> Tuple[Subset, Subset]:
    victim0 = Subset(mex - 1, user_input)
    victim0_as_binary = victim0.big_endian_binary
    victim1 = Subset(mex - 1, victim0_as_binary, hex=False)
    expected = set(victim0._bit_index_array)
    actual = set(victim1._bit_index_array)
    assert (expected == actual)
    return victim0, victim1


def assert_as_full_integers(victim0, victim1):
    expected = victim0.full_integer
    actual = victim1.full_integer
    assert (expected == actual)


def assert_as_hex(victim0, victim1):
    expected = victim0.big_endian_hex
    actual = victim1.big_endian_hex
    assert (expected == actual)

    # brute-force reversal
    leb0 = victim0.big_endian_binary[::-1]
    leb0_i = int(leb0, 2)
    # leb0_h = f'{leb0_i:X}'

    # logarithmic-time reversal
    leh0 = victim0.little_endian_hex
    leh0_i = int(leh0, 16)
    # heh0_h = f'{leh0_i:X}'

    assert leb0_i == leh0_i


def assert_inverted(victim0, victim1):
    expected = ~victim0
    actual = ~victim1
    assert (expected == actual)


def assert_shifted(victim0, victim1):
    expected = victim0 << 12
    actual = victim1 << 12
    assert (expected == actual)
    with pytest.raises(TypeError):
        expected_2 = victim0 >> 12
        actual_2 = victim0 >> 12
        assert expected_2 == actual_2


def assert_from_inv_hex(mex, victim0):
    temp1 = victim0.inv_hex()
    victim3 = Subset(mex - 1, temp1, True)
    expected = ~victim0
    actual = victim3
    assert (expected == actual)


def assert_from_full_integer(mex, victim0):
    temp0 = victim0.full_integer
    victim2 = Subset(max=mex - 1)
    victim2.full_integer = temp0
    expected = victim0
    actual = victim2
    assert (expected == actual)


def assert_bit_count(victim0):
    actual = victim0.bit_count
    expected = count_bits(victim0.big_endian_hex)
    assert (expected == actual)


@given(markers=marker_biggish_array_strategy)
# @settings(max_examples=1000)
@settings(max_examples=g_max_examples, deadline=None)  # for debugging
def test_Subset_from_markers_array(markers):
    assert_for_a_mex(NUM_PLATS_PER_APUC, markers)


@given(sections=sections_smallish_array_strategy)
# @settings(max_examples=1000)
@settings(max_examples=g_max_examples)
def test_Subset_from_sections_array(sections):
    assert_for_a_mex(NSECTIONS, sections)


@given(sections=sections_smallish_array_strategy)
@settings(max_examples=g_max_examples, deadline=None)
def test_smallish_slices(sections):
    victim = Subset(max=15, user_input=sections)
    victim_slices = victim.little_endian_slices
    recon = Subset(max=15)
    recon.little_endian_slices = victim_slices
    assert victim == recon


@given(markers=marker_biggish_array_strategy)
@settings(max_examples=g_max_examples)
def test_biggish_slices(markers):
    victim = Subset(max=NUM_PLATS_PER_APUC - 1, user_input=markers)
    victim_slices = victim.little_endian_slices
    recon = Subset(max=NUM_PLATS_PER_APUC - 1)
    recon.little_endian_slices = victim_slices
    assert victim == recon


@given(sections=sections_smallish_array_strategy)
@settings(max_examples=g_max_examples)
def test_section_indicator(sections):
    r"""build a VECTOR of all-ones from any random 'sections' mask"""
    temp0 = Subset(max=15, user_input=sections)
    temp1 = ~temp0
    temp3 = temp0.list + temp1.list
    temp2 = Subset(max=15, user_input=temp3)
    assert numpy.all(temp2.little_endian_numpy_bool_array)


@given(wl_bools=wordline_booleans_strategy)
@settings(max_examples=g_max_examples)
def test_wordline_from_bools(wl_bools):
    accum = 0
    for i, b in enumerate(wl_bools):
        if b:
            accum += 2**i
    temp0 = Subset(max=NUM_PLATS_PER_APUC - 1, user_input=wl_bools)
    # debug_me = temp0.big_endian_hex
    fi = temp0.full_integer
    assert accum == fi


def test_little_endian_hex():
    t0 = Mask("0001")
    t1 = t0.little_endian_hex
    assert t1 == "8000"

    assert Mask("0002").little_endian_hex == "4000"
    assert Mask("0003").little_endian_hex == "C000"
    assert Mask("1000").little_endian_hex == "0008"
    assert Mask("0500").little_endian_hex == "00A0"

    bigger_case = Subset(max=31, user_input="10000000")
    bigger_case_actual = bigger_case.little_endian_hex
    assert bigger_case_actual == "00000008"

    bigger_case_1 = Subset(max=63, user_input="1000000000000000")
    bigger_case_actual_1 = bigger_case_1.little_endian_hex
    assert bigger_case_actual_1 == "0000000000000008"

    bigger_case_2 = Subset(max=127, user_input="10000000000000000000000000000000")
    bigger_case_actual_2 = bigger_case_2.little_endian_hex
    assert bigger_case_actual_2 == "00000000000000000000000000000008"

    bigger_case_3 = Subset(max=255, user_input="1" + 63 * "0")
    bigger_case_actual_3 = 63 * "0" + "8"
    assert bigger_case_3.little_endian_hex == bigger_case_actual_3

    bigger_case_3 = Subset(max=511, user_input="1" + 127 * "0")
    bigger_case_actual_3 = 127 * "0" + "8"
    assert bigger_case_3.little_endian_hex == bigger_case_actual_3

    bigger_case_4 = Subset(max=1023, user_input="1" + 255 * "0")
    bigger_case_actual_4 = 255 * "0" + "8"
    assert bigger_case_4.little_endian_hex == bigger_case_actual_4

    bigger_case_5 = Subset(max=NUM_PLATS_PER_APUC - 1,
                           user_input="1" + (512 * NUM_HALF_BANKS_PER_APUC - 1) * "0")
    bigger_case_actual_5 = (512 * NUM_HALF_BANKS_PER_APUC - 1) * "0" + "8"
    assert bigger_case_5.little_endian_hex == bigger_case_actual_5

