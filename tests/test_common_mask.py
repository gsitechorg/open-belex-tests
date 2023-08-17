r"""
By Brian Beckman
"""

from hypothesis import given, settings

import numpy

import pytest

from open_belex.common.constants import NSECTIONS
from open_belex.common.mask import IllegalArgumentError, Mask
from open_belex_tests.strategies import mask_hex_strategy


@settings(deadline=None)
@given(mask_hex=mask_hex_strategy)
def test_Mask_class(mask_hex):
    mask_asserts(mask_hex)


def mask_asserts(mask_hex):

    msk = Mask(mask_hex)

    # test iteration by index

    for i in range(len(msk)):
        m = msk.list[i]
        assert type(m) is numpy.int16
        assert (m >= 0) and (m <= (NSECTIONS - 1))

    # test iteration by iterator; accumulate in a set.

    first_iteration = set([])
    for m in msk:
        first_iteration.add(m)
        assert type(m) is numpy.int16
        assert (m >= 0) and (m <= (NSECTIONS - 1))

    # Check that iterator resets to zero and that list has no duplicates.

    second_iteration = []
    for m in msk:
        second_iteration.append(m)
    assert first_iteration == set(second_iteration)
    assert len(first_iteration) == len(msk)

    # Check conversion to list.

    assert second_iteration == list(msk)
    assert second_iteration == list(msk.array)

    # Check zero property

    temp = msk.is_zero
    if mask_hex == '0000':
        assert temp
    else:
        assert (not temp)

    # Check hex property

    as_bex = msk.big_endian_hex
    assert as_bex == mask_hex

    # Check binary property

    as_beb = msk.big_endian_binary
    nu0 = Mask(as_beb, as_hex=False)
    result = (msk == nu0)
    assert result

    # Check inversion

    nu = ~msk
    result = (nu == msk)
    assert (not result)

    mu = ~nu
    result = (mu == msk)
    assert result

    # Check list, slice, and array properties

    def implies(a: bool, b: bool) -> bool:
        result = b or (not a)
        return result

    def eqv(a: bool, b: bool) -> bool:
        result = implies(a, b) and implies(b, a)
        return result

    as_list = msk.list
    as_array = msk.array
    as_bools = msk.little_endian_numpy_bool_array
    as_slices = msk.little_endian_slices
    # An empty list implies an empty array and vice versa.
    assert eqv( (len(as_list) == 0), (as_array.size == 0) )
    # An empty list is equivalent to the "is_zero" property.
    assert eqv( (len(as_list) == 0), msk.is_zero)
    # An empty list implies all False as bools and vice versa.
    assert eqv( (len(as_list) == 0), (not numpy.any(as_bools)) )
    # An empty list implies no slices and vice versa.
    assert eqv( (len(as_list) == 0), (len(as_slices) == 0) )
    # Empty or not, as list equals list rep of the array.
    assert as_list == list(as_array)
    # As list equals the private, internal, base representation.
    assert numpy.all(as_list == msk._bit_index_array)
    # Each bit index is equivalent to the corresponding boolean.
    drop_by_drop = [i in as_list for i in range(NSECTIONS)]
    assert numpy.all(drop_by_drop == as_bools)
    # All slices have no step attribute.
    assert numpy.all([slice.step == None for slice in as_slices])

    # Check bounds

    assert msk.max == NSECTIONS - 1
    assert msk.mex == NSECTIONS
    assert msk.max_value == 0xFFFF

    # Check the '&' and '==' operators (temporaries as debugger stops):

    assert numpy.all((msk & msk) == msk)
    t1 = msk
    t2 = ~msk
    test0 = msk & ~msk
    assert (msk & ~msk).is_zero

    # Special cases from README.md

    msk2 = Mask([3, 4, 12, 13, 14])
    inv_msk2 = Mask("8FE7")
    assert (msk2 & inv_msk2).is_zero
    assert (msk2 == ~inv_msk2)
    assert (msk2 == ~~msk2)

    assert msk2[0] == False
    assert msk2[3] == True
    assert msk2.list[0] == 3  # little endian!

    assert Mask("400D") == Mask([0, 2, 3, 14])
    assert Mask("400D") == Mask({0, 2, 3, 14})


def test_Mask_kwargs():
    temp1 = Mask([0, 2, 3, 14], as_hex=False)
    temp0 = Mask(bit_indices=[0, 2, 3, 14])
    assert temp0 == temp1
    temp2 = Mask()
    temp2.full_integer = 0x400D
    assert (temp2
            == Mask(bit_numbers=[0, 2, 3, 14])
            # Order doesn't matter
            == Mask(bit_numbers=[14, 3, 2, 0])
            # Initialization from a set
            == Mask({0, 2, 3, 14})
            # Duplicates are allowed
            == Mask(bit_indices=[3, 2, 2, 0, 3, 3, 14, 0, 0, 0])
            == Mask(packed_integer=0x400D)
            == Mask(packed_integer=16397)
            == Mask(packed_integer=(2 ** 0 | 2 ** 2 | 2 ** 3 | 2 ** 14))
            == Mask(hex_str=f'{(2 ** 0 | 2 ** 2 | 2 ** 3 | 2 ** 14):X}')
            == Mask(hex_str='400D')
            == Mask(hex_str='0x400D')
            == Mask(hex_str='400d')
            == Mask(hex_str='0x400d')
            == Mask(hex_str='40_0D')
            == Mask(hex_str='0x40_0D')
            == Mask(hex_str='40_0d')
            == Mask(hex_str='0x40_0d')
            == Mask(packed_integer=0b0100_0000_0000_1101)
            == Mask(packed_integer=0b100_0000_0000_1101)
            == Mask(binary_str='0100000000001101')
            == Mask(binary_str='0b0100000000001101')
            == Mask(binary_str='0b0100_0000_0000_1101')
            == Mask(binary_str='100000000001101')
            == Mask(binary_str='0b100000000001101')
            == Mask(binary_str='0b100_0000_0000_1101')
            )

    with pytest.raises(ValueError):
        foo = (temp2 << -1)
    assert (temp2 <<  0) == temp2
    assert (temp2 <<  1) == Mask("801A")
    assert (temp2 <<  2) == Mask("0034")
    assert (temp2 <<  3) == Mask("0068")
    assert (temp2 <<  4) == Mask("00D0")
    assert (temp2 <<  5) == Mask("01A0")
    assert (temp2 <<  6) == Mask("0340")
    assert (temp2 <<  7) == Mask("0680")
    assert (temp2 <<  8) == Mask("0D00")
    assert (temp2 <<  9) == Mask("1A00")
    assert (temp2 << 10) == Mask("3400")
    assert (temp2 << 11) == Mask("6800")
    assert (temp2 << 12) == Mask("D000")
    assert (temp2 << 13) == Mask("A000")
    assert (temp2 << 14) == Mask("4000")
    assert (temp2 << 15) == Mask("8000")
    with pytest.raises(ValueError):
        foo = temp2 << 16

    temp3 = Mask(binary_str='1101')
    assert temp3 == Mask('000D')

    temp4 = Mask()
    temp4.full_integer = 22461
    assert (temp4
            == Mask(packed_integer=0x57bd)
            == Mask(packed_integer=0b_0101_0111_1011_1101)
            == Mask(bit_indices=[14, 12, 10, 9, 8, 7, 5, 4, 3, 2, 0])
            )


def test_Mask_kwargs_exceptions():

    with pytest.raises(ValueError) as e:
        Mask(42)
    print()
    print(e.value)

    with pytest.raises(IllegalArgumentError) as e:
        Mask(42, bit_indices=[0])
    print()
    print(e.value)

    with pytest.raises(IllegalArgumentError) as e:
        Mask(user_input=42, as_hex=False, bit_indices=[0])
    print()
    print(e.value)

    # Weird, but acceptable
    assert Mask(packed_integer=1) == Mask(as_hex=False, bit_indices=[0])
    assert Mask(packed_integer=1) == Mask(bit_indices=[0], as_hex=False)
    assert Mask(packed_integer=1) == Mask(as_hex=True, bit_indices=[0])
    assert Mask(packed_integer=1) == Mask(bit_indices=[0], as_hex=True)

    with pytest.raises(ValueError) as e:
        Mask(as_hex=False, user_input='400D')
    print()
    print(e.value)

    # Better kwargs overrides user contradiction:
    Mask(as_hex=False, hex_str='400D')
    Mask(as_hex=True, binary_str='100_0000_0000_1011')

    # order doesn't matter with this erstwhile
    # positional argument 'as_hex'.
    Mask(hex_str='400D', as_hex=False)
    Mask(binary_str='100_0000_0000_1011', as_hex=True)


def test_Mask_class_unit():
    mask_asserts('0001')
    mask_asserts('0000')


def fill_assert(power_of_two,
                expected_hex,
                expected_fill_right_hex,
                expected_fill_left_hex):
    temp0 = Mask()

    temp0.full_integer = power_of_two

    assert temp0.big_endian_hex == expected_hex

    assert temp0.flood_filled_right.big_endian_hex == expected_fill_right_hex
    assert temp0.flood_filled_left.big_endian_hex == expected_fill_left_hex


def test_sometimes_order_matters():
    sorted_mask = Mask("FFFF")
    formerly_unknown_sortation = Mask(hex_str='FFFF')
    formerly_reversed_mask = Mask(range(16))
    formerly_unknown_reversed = Mask(bit_indices=range(16))

    assert sorted_mask == formerly_unknown_sortation == \
        formerly_reversed_mask == formerly_unknown_reversed

    sorted_mask_2 = Mask('AAAA')
    formerly_unknown_sortation_2 = Mask(hex_str='AAAA')
    formerly_reversed_mask_2 = Mask(range(1, 16, 2))
    formerly_unknown_reversed_2 = Mask(bit_indices=range(1, 16, 2))

    assert sorted_mask_2 == formerly_unknown_sortation_2 == \
        formerly_reversed_mask_2 == formerly_unknown_reversed_2

    debug_me = 42


def test_full_integer():

    temp0 = Mask()

    temp0.full_integer = 15
    assert temp0 == Mask("000F")
    assert temp0 == Mask([0, 1, 2, 3])
    assert temp0 == Mask([2, 0, 0, 2, 3, 3, 3, 1, 0])
    temp1 = Mask()
    temp1.full_integer = 0xf
    assert temp0 == temp1

    temp0.full_integer = 0

    assert temp0.big_endian_hex == "0000"

    temp1 = temp0.flood_filled_right
    assert temp1 == temp0

    temp2 = temp0.flood_filled_left
    assert temp2 == temp0

    fill_assert(    1, "0001", "0001", "FFFF")
    fill_assert(    2, "0002", "0003", "FFFE")
    fill_assert(    4, "0004", "0007", "FFFC")
    fill_assert(    8, "0008", "000F", "FFF8")
    fill_assert(   16, "0010", "001F", "FFF0")
    fill_assert(   32, "0020", "003F", "FFE0")
    fill_assert(   64, "0040", "007F", "FFC0")
    fill_assert(  128, "0080", "00FF", "FF80")
    fill_assert(  256, "0100", "01FF", "FF00")
    fill_assert(  512, "0200", "03FF", "FE00")
    fill_assert( 1024, "0400", "07FF", "FC00")
    fill_assert( 2048, "0800", "0FFF", "F800")
    fill_assert( 4096, "1000", "1FFF", "F000")
    fill_assert( 8192, "2000", "3FFF", "E000")
    fill_assert(16384, "4000", "7FFF", "C000")
    fill_assert(32768, "8000", "FFFF", "8000")

    for _ in range(100):
        prng = numpy.random.default_rng()
        temp1 = prng.integers(low=0, high=0xffff)
        temp0.full_integer = temp1
        assert temp0.big_endian_hex == f'{temp1:04X}'



NIBBLE_COUNT_TABLE = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]


def count_bits(value: int) -> int:
    if (value < 0) or (value > 0xFFFF):
        raise ValueError(
            f"Count_bits only works on integers greater than 0 and less than 65535;"
            f" you gave {value}")
    result = 0
    result += NIBBLE_COUNT_TABLE[(value &    0xF) >>  0]
    result += NIBBLE_COUNT_TABLE[(value &   0xF0) >>  4]
    result += NIBBLE_COUNT_TABLE[(value &  0xF00) >>  8]
    result += NIBBLE_COUNT_TABLE[(value & 0xF000) >> 12]
    return result


def test_count_bits():
    assert count_bits(    0) ==  0
    assert count_bits(    1) ==  1
    assert count_bits(    2) ==  1
    assert count_bits(    4) ==  1
    assert count_bits(    8) ==  1
    assert count_bits(   15) ==  4
    assert count_bits(   16) ==  1
    assert count_bits(   32) ==  1
    assert count_bits(   64) ==  1
    assert count_bits(  128) ==  1
    assert count_bits(  255) ==  8
    assert count_bits(  256) ==  1
    assert count_bits(  512) ==  1
    assert count_bits( 1024) ==  1
    assert count_bits( 2048) ==  1
    assert count_bits( 4095) == 12
    assert count_bits( 4096) ==  1
    assert count_bits( 8192) ==  1
    assert count_bits(16384) ==  1
    assert count_bits(32768) ==  1
    assert count_bits(65535) == 16

    with pytest.raises(ValueError) as e:
        count_bits(-1)

    with pytest.raises(ValueError) as e:
        count_bits(65536)

    temp0 = Mask()
    for _ in range(100):
        prng = numpy.random.default_rng()
        temp1 = prng.integers(low=0, high=0xffff)
        temp0.full_integer = temp1
        debug_me = 42
        assert temp0.bit_count == count_bits(temp1)


def test_Mask_is_idempotent():
    temp0 = Mask("0038")
    temp1 = Mask([3, 4, 5])
    temp2 = Mask(temp1)
    debug_me = 42
    assert temp2 == temp0


def test_Mask_indexing():
    temp0 = Mask("0038")
    temp1 = [3, 4, 5]
    for i in range(NSECTIONS):
        if i in temp1:
            assert temp0[i]
        else:
            assert not temp0[i]
