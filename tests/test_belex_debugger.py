r"""
By Dylon Edwards
"""

import numpy as np

from open_belex.diri.half_bank import DIRI
from open_belex.literal import RL, VR, belex_apl

from open_belex_tests.utils import parameterized_belex_test


@belex_apl
def debug_me(Belex, dst: VR, src: VR):
    # NOTE: Feel free to set a break point here

    # Belex.assert_true(.) will only be evaluated within a debug context
    Belex.assert_true(src.row_number is not None)

    # This assertion form accepts a boolean value
    Belex.assert_true(Belex.glass(src, plats=32) == '\n'.join([
        '[A C 2 A 8 5 D 7 B 2 8 9 9 9 E 8 5 6 5 6 6 A C 0 2 7 4 5 7 8 9 D]',
        '[B 4 1 3 7 7 5 F F D 1 0 B F 1 D A 3 9 2 8 6 C A 2 B 7 5 4 A 2 D]',
        '[B 3 F 1 B 1 1 2 E 1 7 C 7 6 E D 1 5 E E 2 9 0 C 7 9 B B D B 3 C]',
        '[8 A 1 2 5 A 0 5 D 2 5 4 F 3 B 8 A 5 4 4 F 9 C 4 A A 0 9 E 4 7 B]',
    ]))

    # This assertion form uses a closure that will only be invoked within a debug context
    Belex.assert_true(lambda: Belex.diri.glass(src.row_number, plats=32) == '\n'.join([
        '00101001101011100100110000010011',
        '01010100010011101000100011001011',
        '00001001101110000111101100001110',
        '11000100100010111000111011011001',
        '11111110011010011100010011111110',
        '11101001101011100011100010110110',
        '00100000101111110111000110001001',
        '10101000100100110011010101111101',
        '10111111111011110110000001110001',
        '10011101100011001101010111100110',
        '01001111110001010000011000111001',
        '10000001110011011010101101000101',
        '00000111100111001010000001011011',
        '10110001110000100101110011001000',
        '01000111000000101111101001111001',
        '11011010101111110000011000000111',
    ]))

    tmp = Belex.VR()

    RL[::] <= src()
    Belex.assert_true(Belex.glass(RL) == Belex.glass(src))

    tmp[::] <= RL()
    Belex.assert_true(Belex.glass(tmp) == Belex.glass(RL))

    RL[::] <= 0
    RL[::] <= tmp()
    dst[::] <= RL()

    Belex.assert_true(lambda: np.array_equal(Belex.diri.hb[src.row_number],
                                             Belex.diri.hb[dst.row_number]))


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_debug_me(diri: DIRI):
    src_vp = 0
    dst_vp = 1
    assert not np.array_equal(diri.hb[src_vp], diri.hb[dst_vp])
    debug_me(dst_vp, src_vp)
    assert np.array_equal(diri.hb[src_vp], diri.hb[dst_vp])
