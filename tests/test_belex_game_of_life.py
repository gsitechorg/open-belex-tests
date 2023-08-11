r"""By Dylon Edwards and Brian Beckman

Copyright 2019 - 2023 GSI Technology, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from collections import deque

from open_belex.diri.half_bank import DIRI
from open_belex.literal import ERL, NRL, RL, SRL, VR, WRL, belex_apl

from open_belex_libs.game_of_life import \
    gosper_gun_one_period  # TODO: move from lib to here
from open_belex_libs.game_of_life import (
    fa4, gol_in_section_danilan, gol_in_section_danilan_2,
    gol_in_section_danilan_2_manually_inlined_and_laned,
    gol_in_section_defactored, gol_in_section_refactored,
    gosper_glider_gun_tutorial, gosper_gun_write_initial_pattern,
    gvrc_eq_imm_16_msk)

from open_belex_tests.utils import parameterized_belex_test


def pdisplay(string: str, message=""):
    print(message)
    print(string.replace("0", "."))


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#  G A M E   O F   L I F E   T U T O R I A L
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


# @pytest.mark.skip("For tutorial live-coding session")
@parameterized_belex_test
def test_game_of_life_tutorial(diri: DIRI):
    petri_dish = 0
    gosper_glider_gun_tutorial(petri_dish)  # in belex-libs
    return


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#  O N E   G E N E R A T I O N ,   T W O   A L G O R I T H M S
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


@belex_apl
def gol_in_section_refactored_one_gen(Belex, petri_dish: VR):
    """Refactored to call library routine fa4 in belex-libs."""

    gosper_gun_write_initial_pattern(petri_dish)

    number_of_plats_to_glass = 40
    glass_kwargs = {"plats": number_of_plats_to_glass,
                    "sections": 16, "fmt": "bin", "order": "lsb"}
    actual_rows = Belex.glass(petri_dish, **glass_kwargs)
    expected_rows = "\n".join(s.replace(".", "0") for s in [
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . 1 1 . . . . . . 1 1 . . . . . . . . . . . . 1 1 . . .]",
        "[. . . . . . . . . . . . 1 . . . 1 . . . . 1 1 . . . . . . . . . . . . 1 1 . . .]",
        "[. 1 1 . . . . . . . . 1 . . . . . 1 . . . 1 1 . . . . . . . . . . . . . . . . .]",
        "[. 1 1 . . . . . . . . 1 . . . 1 . 1 1 . . . . 1 . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . 1 . . . . . 1 . . . . . . . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . 1 . . . 1 . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]", ])
    Belex.assert_true(actual_rows == expected_rows)

    RL[::] <= petri_dish()
    # spot check
    actual_rows = Belex.glass(RL, **glass_kwargs)
    expected_rows = "\n".join(s.replace(".", "0") for s in [
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . 1 1 . . . . . . 1 1 . . . . . . . . . . . . 1 1 . . .]",
        "[. . . . . . . . . . . . 1 . . . 1 . . . . 1 1 . . . . . . . . . . . . 1 1 . . .]",
        "[. 1 1 . . . . . . . . 1 . . . . . 1 . . . 1 1 . . . . . . . . . . . . . . . . .]",
        "[. 1 1 . . . . . . . . 1 . . . 1 . 1 1 . . . . 1 . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . 1 . . . . . 1 . . . . . . . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . 1 . . . 1 . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]", ])
    Belex.assert_true(actual_rows == expected_rows)

    u0 = Belex.VR(0)
    u1 = Belex.VR(0)
    u2 = Belex.VR(0)
    u3 = Belex.VR(0)

    fa4(u0, u1, u2, u3, petri_dish)

    def check_row(u, row):
        actual_row = Belex.glass(u, plats=number_of_plats_to_glass,
                                 sections=[4], fmt="bin", order="lsb")
        if Belex.debug:
            actual_row = actual_row.split("\n")[4]
        expected_row = "\n".join(s.replace(".", "0") for s in [row])
        Belex.assert_true(actual_row == expected_row)

    check_row(
        u0,
        '[1 . . 1 . . . . . . 1 . . 1 . . 1 . 1 . 1 1 1 1 . . . . . . . . . . . 1 1 . . .]')
    check_row(
        u1,
        '[. 1 1 . . . . . . . . 1 1 1 1 1 . 1 . . 1 . . 1 . . . . . . . . . . 1 1 1 1 . .]')
    check_row(
        u2,
        '[. . . . . . . . . . . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . .]')
    check_row(
        u3,
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]')

    new_gen = Belex.VR(0)

    # If it's ON and has 2 or 3 neighbors, leave it ON.
    RL[::] <= petri_dish()  # it's ON
    RL[::] &= u1()  # bit one (coefficient of 2) is ON
    RL[::] &= ~u2()  # bit two (coefficient of 4) is OFF
    RL[::] &= ~u3()  # bit tre (coefficient of 8) is OFF

    new_gen[::] <= RL()

    # If it's OFF and has exactly 3 neighbors, turn it ON.
    RL[::] <= 1
    RL[::] &= ~petri_dish()  # it's OFF
    RL[::] &= u0()  # bit zro (coefficient of 0) is ON
    RL[::] &= u1()  # bit one (coefficient of 2) is ON
    RL[::] &= ~u2()  # bit two (coefficient of 4) is OFF
    RL[::] &= ~u3()  # bit tre (coefficient of 8) is OFF

    new_gen[::] |= RL()

    actual_rows = Belex.glass(new_gen, **glass_kwargs)
    expected_rows = "\n".join(s.replace(".", "0") for s in [
        # plat numbers:       1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3
        # 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . 1 . . . . . . . 1 . 1 . . . . . . . . . . . 1 1 . . .]',
        '[. . . . . . . . . . . . 1 1 . . . . . . 1 . . 1 . . . . . . . . . . . 1 1 . . .]',
        '[. 1 1 . . . . . . . . 1 1 . . . . 1 1 . . 1 . 1 . . . . . . . . . . . . . . . .]',
        '[. 1 1 . . . . . . . 1 1 1 . . . . 1 1 . . . 1 . 1 . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . 1 1 . . . . 1 1 . . . . . 1 . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]'])
    Belex.assert_true(actual_rows == expected_rows)


@belex_apl
def gol_in_section_defactored_one_gen(Belex, petri_dish: VR):
    r"""Full-adder / half-adder solution (not Dan Ilan's), inline."""

    gosper_gun_write_initial_pattern(petri_dish)

    number_of_plats_to_glass = 40
    glass_kwargs = {"plats": number_of_plats_to_glass,
                    "sections": 16, "fmt": "bin", "order": "lsb"}
    actual_rows = Belex.glass(petri_dish, **glass_kwargs)
    expected_rows = "\n".join(s.replace(".", "0") for s in [
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . 1 1 . . . . . . 1 1 . . . . . . . . . . . . 1 1 . . .]",
        "[. . . . . . . . . . . . 1 . . . 1 . . . . 1 1 . . . . . . . . . . . . 1 1 . . .]",
        "[. 1 1 . . . . . . . . 1 . . . . . 1 . . . 1 1 . . . . . . . . . . . . . . . . .]",
        "[. 1 1 . . . . . . . . 1 . . . 1 . 1 1 . . . . 1 . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . 1 . . . . . 1 . . . . . . . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . 1 . . . 1 . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]", ])
    Belex.assert_true(actual_rows == expected_rows)

    RL[::] <= petri_dish()
    # spot check
    actual_rows = Belex.glass(RL, **glass_kwargs)
    expected_rows = "\n".join(s.replace(".", "0") for s in [
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . 1 1 . . . . . . 1 1 . . . . . . . . . . . . 1 1 . . .]",
        "[. . . . . . . . . . . . 1 . . . 1 . . . . 1 1 . . . . . . . . . . . . 1 1 . . .]",
        "[. 1 1 . . . . . . . . 1 . . . . . 1 . . . 1 1 . . . . . . . . . . . . . . . . .]",
        "[. 1 1 . . . . . . . . 1 . . . 1 . 1 1 . . . . 1 . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . 1 . . . . . 1 . . . . . . . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . 1 . . . 1 . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]", ])
    Belex.assert_true(actual_rows == expected_rows)

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    #  V o n - N e u m a n n   b i t s   ( n o n - d i a g o n a l )
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

    x1 = w = Belex.VR(0)  # x1 is an alias, helpful for correctly writing the adder
    w[::] <= WRL()
    # spot check
    actual_rows = Belex.glass(w, **glass_kwargs)
    expected_rows = "\n".join(s.replace(".", "0") for s in [
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . 1 1 . . . . . . 1 1 . . . . . . . . . . . . 1 1 . .]",
        "[. . . . . . . . . . . . . 1 . . . 1 . . . . 1 1 . . . . . . . . . . . . 1 1 . .]",
        "[. . 1 1 . . . . . . . . 1 . . . . . 1 . . . 1 1 . . . . . . . . . . . . . . . .]",
        "[. . 1 1 . . . . . . . . 1 . . . 1 . 1 1 . . . . 1 . 1 . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . 1 . . . . . 1 . . . . . . . 1 . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . 1 . . . 1 . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]", ])
    Belex.assert_true(actual_rows == expected_rows)

    x2 = e = Belex.VR(0)  # x2 is an alias for the adder
    e[::] <= ERL()

    x3 = n = Belex.VR(0)  # x3 is an alias for the adder
    n[::] <= NRL()

    x4 = s = Belex.VR(0)  # x4 is an alias for the adder
    s[::] <= SRL()

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    #  M o o r e   b i t s   ( i n c l u d i n g   d i a g o n a l s )
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

    RL[::] <= w()

    x5 = nw = Belex.VR(0)  # x5 is an alias for the adder
    nw[::] <= NRL()
    # spot check
    actual_rows = Belex.glass(nw, **glass_kwargs)
    expected_rows = "\n".join(s.replace(".", "0") for s in [
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . 1 1 . . . . . . 1 1 . . . . . . . . . . . . 1 1 . .]",
        "[. . . . . . . . . . . . . 1 . . . 1 . . . . 1 1 . . . . . . . . . . . . 1 1 . .]",
        "[. . 1 1 . . . . . . . . 1 . . . . . 1 . . . 1 1 . . . . . . . . . . . . . . . .]",
        "[. . 1 1 . . . . . . . . 1 . . . 1 . 1 1 . . . . 1 . 1 . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . 1 . . . . . 1 . . . . . . . 1 . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . 1 . . . 1 . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]", ])
    Belex.assert_true(actual_rows == expected_rows)

    x6 = sw = Belex.VR(0)  # x6 is an alias for the adder
    sw[::] <= SRL()

    RL[::] <= e()

    x7 = ne = Belex.VR(0)  # x7 is an alias for the adder
    ne[::] <= NRL()

    x8 = se = Belex.VR(0)  # x8 is an alias for the adder
    se[::] <= SRL()

    # +-+-+-+-+
    #  s u m s
    # +-+-+-+-+

    def ha(ssum, cout, a, b):
        """half-adder"""
        RL[::] <= a()
        RL[::] ^= b()
        ssum[::] <= RL()
        RL[::] <= a()
        RL[::] &= b()
        cout[::] <= RL()

    def fa(ssum, cout, a, b, cin, temp):
        """full adder"""
        # ssum = a ^ b ^ cin
        RL[::] <= a()
        RL[::] ^= b()
        RL[::] ^= cin()
        ssum[::] <= RL()
        # cout = (a /\ b) ^ (c /\ (a ^ b))
        RL[::] <= a()
        RL[::] ^= b()
        RL[::] &= cin()
        temp[::] <= RL()
        RL[::] <= a()
        RL[::] &= b()
        RL[::] ^= temp()
        cout[::] = RL()

    def pair_count(s0, s1,  # outputs
                   x1, x2):  # inputs
        r"""Count the number of ON-bits in x1 and x2.
        s0 is the coefficient of 2^0; s1 is the coefficient
        of 2^1; x1 is one input of the pair; x2 is the other
        input of the pair. The truth table of this function
        is just that of the half adder."""
        ha(s0, s1, x1, x2)

    def quad_count(t0, t1, t2,  # 3-bit output
                   sp10, sp20,  # inputs from prior call of pair_count
                   sp11, sp21,  # inputs from prior call of pair_count
                   c0,  # temp for rippled-carry
                   scratch):  #
        r"""Count the 3-bit number of ON-bits in four inputs
        aggregated in four prior calls of pair_count.
        t0, t1, t2 are little-endian outputs; sp10, sp20
        are the 0-bits of two prior pair counts; sp11, sp21
        are the 1-bits of two prior pair counts."""
        ha(t0, c0, sp10, sp20)  # ha5, ha6
        fa(t1, t2, sp11, sp21, c0, scratch)  # fa1, fa2

    def octo_count(u0, u1, u2, u3,  # 4-bit output
                   t10, t20,  # inputs from prior call of quad_count
                   t11, t21,  # inputs from prior call of quad_count
                   t12, t22,  # inputs from prior call of quad_count
                   d0, d1,  # temps for rippled carries
                   scratch):  #
        r"""Count the 4-bit number of ON-bits in eight inputs
        aggregated in two prior calls of quad_count.
        u0, ..., u3 are the little-endian outputs; t10, t20
        are the 0-bits of two prior quad counts; t11, t21
        are the 1-bits of two prior quad counts; t12, t22
        are the 2-bits of two prior quad counts."""
        ha(u0, d0, t10, t20)  # ha7 on the diagram
        fa(u1, d1, t11, t21, d0, scratch)  # fa3
        fa(u2, u3, t12, t22, d1, scratch)  # fa4

    s10 = Belex.VR(0)
    s11 = Belex.VR(0)

    s20 = Belex.VR(0)
    s21 = Belex.VR(0)

    s30 = Belex.VR(0)
    s31 = Belex.VR(0)

    s40 = Belex.VR(0)
    s41 = Belex.VR(0)

    pair_count(s10, s11, x1, x2)
    pair_count(s20, s21, x3, x4)
    pair_count(s30, s31, x5, x6)
    pair_count(s40, s41, x7, x8)

    t10 = Belex.VR(0)
    t11 = Belex.VR(0)
    t12 = Belex.VR(0)

    t20 = Belex.VR(0)
    t21 = Belex.VR(0)
    t22 = Belex.VR(0)

    c0 = Belex.VR(0)
    scratch = Belex.VR(0)

    quad_count(t10, t11, t12,
               s10, s20, s11, s21,
               c0, scratch)
    quad_count(t20, t21, t22,
               s30, s40, s31, s41,
               c0, scratch)

    c1 = Belex.VR(0)

    u0 = Belex.VR(0)
    u1 = Belex.VR(0)
    u2 = Belex.VR(0)
    u3 = Belex.VR(0)

    octo_count(u0, u1, u2, u3,
               t10, t20, t11, t21, t12, t22,
               c0, c1, scratch)

    def check_row(u, row):
        actual_row = Belex.glass(u, plats=number_of_plats_to_glass,
                                 sections=[4], fmt="bin", order="lsb")
        if Belex.debug:
            actual_row = actual_row.split("\n")[4]
        expected_row = "\n".join(s.replace(".", "0") for s in [row])
        Belex.assert_true(actual_row == expected_row)

    check_row(
        u0,
        '[1 . . 1 . . . . . . 1 . . 1 . . 1 . 1 . 1 1 1 1 . . . . . . . . . . . 1 1 . . .]')
    check_row(
        u1,
        '[. 1 1 . . . . . . . . 1 1 1 1 1 . 1 . . 1 . . 1 . . . . . . . . . . 1 1 1 1 . .]')
    check_row(
        u2,
        '[. . . . . . . . . . . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . .]')
    check_row(
        u3,
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]')

    new_gen = Belex.VR(0)

    # If it's ON and has 2 or 3 neighbors, leave it ON.
    RL[::] <= petri_dish()  # it's ON
    RL[::] &= u1()  # bit one (coefficient of 2) is ON
    RL[::] &= ~u2()  # bit two (coefficient of 4) is OFF
    RL[::] &= ~u3()  # bit tre (coefficient of 8) is OFF

    new_gen[::] <= RL()

    # If it's OFF and has exactly 3 neighbors, turn it ON.
    RL[::] <= 1
    RL[::] &= ~petri_dish()  # it's OFF
    RL[::] &= u0()  # bit zro (coefficient of 0) is ON
    RL[::] &= u1()  # bit one (coefficient of 2) is ON
    RL[::] &= ~u2()  # bit two (coefficient of 4) is OFF
    RL[::] &= ~u3()  # bit tre (coefficient of 8) is OFF

    new_gen[::] |= RL()

    actual_rows = Belex.glass(new_gen, **glass_kwargs)
    expected_rows = "\n".join(s.replace(".", "0") for s in [
        # plat numbers:       1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3
        # 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . 1 . . . . . . . 1 . 1 . . . . . . . . . . . 1 1 . . .]',
        '[. . . . . . . . . . . . 1 1 . . . . . . 1 . . 1 . . . . . . . . . . . 1 1 . . .]',
        '[. 1 1 . . . . . . . . 1 1 . . . . 1 1 . . 1 . 1 . . . . . . . . . . . . . . . .]',
        '[. 1 1 . . . . . . . 1 1 1 . . . . 1 1 . . . 1 . 1 . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . 1 1 . . . . 1 1 . . . . . 1 . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]'])
    Belex.assert_true(actual_rows == expected_rows)


@parameterized_belex_test
def test_gol_in_section_simplified_one_generation(diri: DIRI):
    r"""Full-adder / half-adder solution (not Dan Ilan's), inline."""

    petri_dish = 0

    gol_in_section_defactored_one_gen(petri_dish)
    gol_in_section_refactored_one_gen(petri_dish)
    return


# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#   R E F A C T O R E D ,   O N E   P E R I O D
# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-


THIRTIETH_GEN = \
    "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]\n" \
    "[. . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . .]\n" \
    "[. . . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . .]\n" \
    "[. . . . . . . . . . . . . 1 1 . . . . . . 1 1 . . . . . . . . . . . . 1 1 . . .]\n" \
    "[. . . . . . . . . . . . 1 . . . 1 . . . . 1 1 . . . . . . . . . . . . 1 1 . . .]\n" \
    "[. 1 1 . . . . . . . . 1 . . . . . 1 . . . 1 1 . . . . . . . . . . . . . . . . .]\n" \
    "[. 1 1 . . . . . . . . 1 . . . 1 . 1 1 . . . . 1 . 1 . . . . . . . . . . . . . .]\n" \
    "[. . . . . . . . . . . 1 . . . . . 1 . . . . . . . 1 . . . . . . . . . . . . . .]\n" \
    "[. . . . . . . . . . . . 1 . . . 1 . . . . . . . . . . . . . . . . . . . . . . .]\n" \
    "[. . . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . .]\n" \
    "[. . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . .]\n" \
    "[. . . . . . . . . . . . . . . . . . . . . . . . . 1 1 . . . . . . . . . . . . .]\n" \
    "[. . . . . . . . . . . . . . . . . . . . . . . . 1 1 . . . . . . . . . . . . . .]\n" \
    "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]\n" \
    "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]\n" \
    "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]"


@belex_apl
def gol_in_section_refactored_one_period(Belex, petri_dish: VR):
    print('\nREfactored GEN 0\n')

    def play(i: int) -> None:
        s = Belex.glass(petri_dish, plats=40, sections=16, fmt="bin",
                        order="lsb")
        if s:  # s is None in belex-test
            t = s.replace('0', '.')
            print(t)
            print('\n')
            if i == 29:
                Belex.assert_true(s.replace('0', '.') == THIRTIETH_GEN)

    play(-1)

    for i in range(30):
        gol_in_section_refactored(petri_dish)
        print(f'REfactored GEN {1 + i}\n')
        play(i)


@parameterized_belex_test
def test_gol_in_section_refactored_one_period(diri: DIRI):
    petri_dish = 0
    gosper_gun_write_initial_pattern(petri_dish)
    gol_in_section_refactored_one_period(petri_dish)


# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#   D E F A C T O R E D ,   O N E   P E R I O D
# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-


@belex_apl
def gol_in_section_defactored_one_period(Belex, petri_dish: VR):
    print('\nDEfactored GEN 0\n')

    def play(i: int) -> None:
        s = Belex.glass(petri_dish, plats=40, sections=16, fmt="bin",
                        order="lsb")
        if s:  # s is None in belex-test
            t = s.replace('0', '.')
            print(t)
            print('\n')
            if i == 29:
                Belex.assert_true(s.replace('0', '.') == THIRTIETH_GEN)

    play(-1)

    for i in range(30):
        gol_in_section_defactored(petri_dish)
        print(f'DEfactored GEN {1 + i}\n')
        play(i)


@parameterized_belex_test
def test_gol_in_section_defactored_one_period(diri: DIRI):
    petri_dish = 0
    gosper_gun_write_initial_pattern(petri_dish)
    gol_in_section_defactored_one_period(petri_dish)


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#  D A N   I L A N ' S   F A S T   S O L U T I O N
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


@belex_apl
def gol_in_section_danilan_one_period(Belex, petri_dish: VR):
    print('\nDan Ilan GEN 0\n')

    def play(i: int) -> None:
        s = Belex.glass(petri_dish, plats=40, sections=16, fmt="bin",
                        order="lsb")
        if s:  # it's None in belex-test
            t = s.replace('0', '.')
            print(t)
            print('\n')
            if i == 29:
                Belex.assert_true(s.replace('0', '.') == THIRTIETH_GEN)

    play(-1)

    for i in range(30):
        gol_in_section_danilan(petri_dish)  # in belex_libs
        print(f'Dan Ilan GEN {1 + i}\n')
        play(i)


@parameterized_belex_test
def test_gol_in_section_danilan_one_period(diri: DIRI):
    petri_dish = 0
    gosper_gun_write_initial_pattern(petri_dish)
    gol_in_section_danilan_one_period(petri_dish)


# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#  D a n   I l a n   2 :   f a s t e s t   k n o w n
# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-


@belex_apl
def gol_in_section_danilan_2_one_period(Belex, petri_dish: VR):
    print('\nDan Ilan 2 GEN 0\n')

    def play(i: int) -> None:
        s = Belex.glass(petri_dish, plats=40, sections=16, fmt="bin",
                        order="lsb")
        if s:  # s is None in belex-test
            t = s.replace('0', '.')
            print(t)
            print('\n')
            if i == 29:
                Belex.assert_true(s.replace('0', '.') == THIRTIETH_GEN)

    play(-1)

    for i in range(30):
        gol_in_section_danilan_2(petri_dish)  # in belex_libs
        print(f'Dan Ilan 2 GEN {1 + i}\n')
        play(i)


@parameterized_belex_test
def test_gol_in_section_danilan_2_one_period(diri: DIRI):
    petri_dish = 0
    gosper_gun_write_initial_pattern(petri_dish)
    gol_in_section_danilan_2_one_period(petri_dish)


# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#  D a n I l a n _ 2   m a n u a l l y   l a n e d
# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-


@belex_apl
def gol_in_section_danilan_2_manual_one_period(Belex, petri_dish: VR):
    print('\nDan Ilan 2 manually inlined GEN 0\n')

    def play(i: int) -> None:
        s = Belex.glass(petri_dish, plats=40, sections=16, fmt="bin",
                        order="lsb")
        if s:  # s is None in belex-test
            t = s.replace('0', '.')
            print(t)
            print('\n')
            if i == 29:
                Belex.assert_true(s.replace('0', '.') == THIRTIETH_GEN)

    play(-1)

    for i in range(30):
        gol_in_section_danilan_2_manually_inlined_and_laned(
            petri_dish)  # in belex_libs
        print(f'Dan Ilan 2 GEN {1 + i}\n')
        play(i)


@parameterized_belex_test
def test_gol_in_section_danilan_2_manual_one_period(diri: DIRI):
    petri_dish = 0
    gosper_gun_write_initial_pattern(petri_dish)
    gol_in_section_danilan_2_manual_one_period(petri_dish)



# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#   B R U T E - F O R C E   S O L U T I O N
# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-


# @pytest.mark.skip("takes 16 minutes")
@parameterized_belex_test
def test_gosper_gun_two_generations(diri: DIRI):
    r"""From the old days when the function under test
    had to call Belex.glass."""
    captured_rows = deque()

    petri_dish = 0
    gosper_gun_one_period(petri_dish, captured_glass=captured_rows)

    actual_rows = captured_rows.popleft()
    expected_rows = "\n".join(s.replace(".", "0") for s in [
        # plat numbers:       1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3
        # 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
        # -------------------------------------------------------------------------------
        # . . . . . . . . . . . . . 8 8 . . . . . . 8 8 4 . 6 . . . . . . . . . 8 8 . . .
        # . 6 6 . . . . . . . . E 1 2 2 4 1 E 4 . . 3 3 4 . C . . . . . . . . . 1 1 . . .
        # . 6 6 . . . . . . . . . . 1 2 2 . 1 . . . . . . . . . . . . . . . . . . . . . .
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # -------------------------------------------------------------------------------
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . 1 1 . . . . . . 1 1 . . . . . . . . . . . . 1 1 . . .]",
        "[. . . . . . . . . . . . 1 . . . 1 . . . . 1 1 . . . . . . . . . . . . 1 1 . . .]",
        "[. 1 1 . . . . . . . . 1 . . . . . 1 . . . 1 1 . . . . . . . . . . . . . . . . .]",
        "[. 1 1 . . . . . . . . 1 . . . 1 . 1 1 . . . . 1 . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . 1 . . . . . 1 . . . . . . . 1 . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . 1 . . . 1 . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]",
        "[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]", ])
    assert actual_rows == expected_rows

    actual_rows = captured_rows.popleft()
    expected_rows = "\n".join(s.replace(".", "0") for s in [
        # plat numbers:       1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3
        # 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
        # -------------------------------------------------------------------------------
        # . . . . . . . . . . . . . 8 . . . . . . . 8 4 8 6 . . . . . . . . . . 8 8 . . .
        # . 6 6 . . . . . . . . 4 E F 1 . . E E . 1 2 4 3 C . . . . . . . . . . 1 1 . . .
        # . 6 6 . . . . . . . . . 1 3 . . . . . . . . . . . . . . . . . . . . . . . . . .
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # -------------------------------------------------------------------------------
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . 1 . . . . . . . 1 . 1 . . . . . . . . . . . 1 1 . . .]',
        '[. . . . . . . . . . . . 1 1 . . . . . . 1 . . 1 . . . . . . . . . . . 1 1 . . .]',
        '[. 1 1 . . . . . . . . 1 1 . . . . 1 1 . . 1 . 1 . . . . . . . . . . . . . . . .]',
        '[. 1 1 . . . . . . . 1 1 1 . . . . 1 1 . . . 1 . 1 . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . 1 1 . . . . 1 1 . . . . . 1 . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]'])
    assert actual_rows == expected_rows

    actual_rows = captured_rows.popleft()
    expected_rows = "\n".join(s.replace(".", "0") for s in [
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . 1 1 . . . . . . . 1 . 1 1 . . . . . . . . . . 1 1 . . .]',
        '[. . . . . . . . . . . 1 . 1 . . . . . . 1 1 . 1 1 . . . . . . . . . . 1 1 . . .]',
        '[. 1 1 . . . . . . . 1 . . . . . . 1 1 1 . 1 . 1 1 . . . . . . . . . . . . . . .]',
        '[. 1 1 . . . . . . . 1 . . 1 . . 1 . . 1 . . 1 . 1 . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . 1 . . . . . . 1 1 . . . . 1 . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]',
        '[. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .]'])
    assert actual_rows == expected_rows
    return petri_dish


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#  T A B L E   L O O K U P   ( U N D O N E )
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


@parameterized_belex_test
def test_table_lookup(diri: DIRI):
    # [8:41 AM, 02/28/22] eli ehrman
    c_field = 0x7
    c_imm = 0x2
    inv_imm = ~c_imm & c_field
    # Karnaugh, ternary mask
    # last arg is obviously equiv to c_field but input in this
    # form waiting for GL bug fix
    output_vp = 0
    input_vp = 1

    gvrc_eq_imm_16_msk(
        flags=output_vp,
        dest_mrk=0x4,
        src=input_vp,
        imm_test=c_imm,
        inv_mask=inv_imm)
