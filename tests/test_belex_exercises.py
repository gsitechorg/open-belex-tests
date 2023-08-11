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

from open_belex.common.constants import (NUM_HALF_BANKS_PER_APUC,
                                         NUM_PLATS_PER_HALF_BANK)
from open_belex.diri.half_bank import DIRI
from open_belex.literal import (GL, INV_GL, INV_RL, NRL, RL, VR, WRL, Mask,
                                Section, apl_commands, belex_apl)
from open_belex.utils.example_utils import convert_to_u16

from open_belex_libs.common import cpy_imm_16_to_rl
from open_belex_libs.tartan import (walk_marks_eastward,
                                    write_markers_in_plats_matching_value,
                                    write_to_marked)

from open_belex_tests.utils import parameterized_belex_test

#   ___  _    _   ___ _    ___ ___ ___ ___   _____       _
#  / _ \| |__| | | _ ) |  | __/ __/ __|_ _| |_   _|__ __| |_ ___
# | (_) | / _` | | _ \ |__| _| (_| (__ | |    | |/ -_|_-<  _(_-<
#  \___/|_\__,_| |___/____|___\___\___|___|   |_|\___/__/\__/__/


# +-+-+-+-+-+-+-+-+ +-+
# |E|X|E|R|C|I|S|E| |2|
# +-+-+-+-+-+-+-+-+ +-+


@belex_apl
def exercise_2(Belex, sb: VR):

    # BELEX takes 'sb', being the first parameter of this FUT
    # (function under test), implicitly as containing the actual
    # values to check against expected values in the C-sim. The
    # expected values are in the SB returned from the test
    # function, 'test_exercise_2', below. DIRI computes the
    # expected values, so DIRI is the "ground truth" for C-sim.

    os = "0x0001"
    fs = "0xFFFF"

    RL[fs] <= 0
    RL[os] <= 1
    sb[os] <= RL()


@parameterized_belex_test
def test_exercise_2(diri: DIRI):
    sb = 7
    exercise_2(sb)
    assert all(convert_to_u16(diri.hb[sb]) == 0x0001)
    return sb  # expected values in SB[7]


# +-+-+-+-+-+-+-+-+ +-+ +-+-+-+-+-+
# |E|X|E|R|C|I|S|E| |3| |M|U|L|T|I|
# +-+-+-+-+-+-+-+-+ +-+ +-+-+-+-+-+


@belex_apl
def exercise_3_multi(Belex, sb: VR, sb2: VR):
    """Write the complement of [secs 0,4,8,12] to [secs 1,5,9,13] of
    multiple VRs, using multi-SB syntax on the left-hand side."""
    fs = "0xFFFF"
    ts = "0x2222"
    RL[fs] <= 0  # clear RL
    msk = Belex.Mask(ts)
    msk[sb, sb2] <= ~NRL()
    # 'sb', being the first parameter, contains actual values for C-sim


@parameterized_belex_test
def test_exercise_3_multi(diri: DIRI):
    sb = 1  # pick VRs randomly
    sb2 = 2
    exercise_3_multi(sb, sb2)
    actual_sb = diri.glass(sb, plats=4, sections=16)
    assert actual_sb == '0000\n1111\n0000\n0000\n' \
                        '0000\n1111\n0000\n0000\n' \
                        '0000\n1111\n0000\n0000\n' \
                        '0000\n1111\n0000\n0000'
    actual_sb2 = diri.glass(sb2, plats=4, sections=16)
    assert actual_sb2 == '0000\n1111\n0000\n0000\n' \
                         '0000\n1111\n0000\n0000\n' \
                         '0000\n1111\n0000\n0000\n' \
                         '0000\n1111\n0000\n0000'
    return sb  # expected values from DIRI into C-sim.


@belex_apl
def exercise_3_multi_via_shift(Belex, sb: VR, sb2: VR):
    """Write the complement of [secs 0,4,8,12] to [secs 1,5,9,13] of
    multiple VRs, using multi-SB syntax on the left-hand side."""
    fs = "0xFFFF"
    ts = "0x1111"
    RL[fs] <= 0  # clear RL
    msk = Belex.Mask(ts)
    (msk << 1)[sb, sb2] <= ~NRL()
    # 'sb', being the first parameter, contains actual values for C-sim


@parameterized_belex_test
def test_exercise_3_via_shift_multi(diri: DIRI):
    sb = 1  # pick VRs randomly
    sb2 = 2
    exercise_3_multi_via_shift(sb, sb2)
    actual_sb = diri.glass(sb, plats=4, sections=16)
    assert actual_sb == '0000\n1111\n0000\n0000\n' \
                        '0000\n1111\n0000\n0000\n' \
                        '0000\n1111\n0000\n0000\n' \
                        '0000\n1111\n0000\n0000'
    actual_sb2 = diri.glass(sb2, plats=4, sections=16)
    assert actual_sb2 == '0000\n1111\n0000\n0000\n' \
                         '0000\n1111\n0000\n0000\n' \
                         '0000\n1111\n0000\n0000\n' \
                         '0000\n1111\n0000\n0000'
    return sb  # expected values from DIRI into C-sim.


@belex_apl
def exercise_3_multi_via_shift_2(Belex, sb: VR, sb2: VR):
    """Write the complement of [secs 0,4,8,12] to [secs 1,5,9,13] of
    multiple VRs, using multi-SB syntax on the left-hand side."""
    fs = "0xFFFF"
    ts = "0x1111"
    RL[fs] <= 0  # clear RL
    msk = Belex.Mask(ts) << 1
    msk[sb, sb2] <= ~NRL()
    # 'sb', being the first parameter, contains actual values for C-sim


@parameterized_belex_test
def test_exercise_3_via_shift_multi_2(diri: DIRI):
    sb = 1  # pick VRs randomly
    sb2 = 2
    exercise_3_multi_via_shift_2(sb, sb2)
    actual_sb = diri.glass(sb, plats=4, sections=16)
    assert actual_sb == '0000\n1111\n0000\n0000\n' \
                        '0000\n1111\n0000\n0000\n' \
                        '0000\n1111\n0000\n0000\n' \
                        '0000\n1111\n0000\n0000'
    actual_sb2 = diri.glass(sb2, plats=4, sections=16)
    assert actual_sb2 == '0000\n1111\n0000\n0000\n' \
                         '0000\n1111\n0000\n0000\n' \
                         '0000\n1111\n0000\n0000\n' \
                         '0000\n1111\n0000\n0000'
    return sb  # expected values from DIRI into C-sim.


# +-+-+-+-+-+-+-+-+ +-+
# |E|X|E|R|C|I|S|E| |3|
# +-+-+-+-+-+-+-+-+ +-+


@belex_apl
def exercise_3(Belex, sb: VR):
    """Write the complement of [secs 0,4,8,12] to [secs 1,5,9,13]."""
    fs = "0xFFFF"
    ts = "0x2222"
    RL[fs] <= 0  # clear RL
    sb[ts] <= ~NRL()  # write 0, 4, 8, 12 into 1, 5, 9, 13 of 'sb'
    # 'sb', being the first parameter, contains actual values for C-sim


@parameterized_belex_test
def test_exercise_3(diri: DIRI):
    sb = 1  # pick a VR randomly
    exercise_3(sb)
    actual_sb = diri.glass(sb, plats=4, sections=16)
    assert actual_sb == '0000\n1111\n0000\n0000\n' \
                        '0000\n1111\n0000\n0000\n' \
                        '0000\n1111\n0000\n0000\n' \
                        '0000\n1111\n0000\n0000'
    return sb  # expected values from DIRI into C-sim.


# +-+-+-+-+-+-+-+-+ +-+
# |E|X|E|R|C|I|S|E| |4|
# +-+-+-+-+-+-+-+-+ +-+


@belex_apl
def exercise_4(Belex, out: VR):
    """[a] Repeat [Exercise] 2 [write 1 to 1:SB[0]].
    [b] Shift [sec 0] to West (i.e. read from E)
    -- that is probably a typo, and the following was probably meant
    [b'] Shift [sec 0] to East, i.e., read from W
    - Check RSP is still 1
    - Invert as in Exercise 3
    - Check the complement is also 1"""

    # [a] Repeat [Exercise] 2 [write 1 to 1:SB[0]].
    cpy_imm_16_to_rl("0x0001")
    out[::] <= RL()

    # [b'] Shift [sec 0] to East, i.e., read from W
    out["0x0001"] <= WRL()

    # - Invert as in Exercise 3
    RL["0xFFFF"] <= 0
    RL["0x0001"] <= out()
    out["0x0002"] <= ~NRL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_exercise_4(diri: DIRI) -> int:
    out = 0
    exercise_4(out)
    actual_output = convert_to_u16(diri.hb[out])
    for hb in range(NUM_HALF_BANKS_PER_APUC):
        i = hb * NUM_PLATS_PER_HALF_BANK + 0
        assert actual_output[i] == 0x0002
        i = hb * NUM_PLATS_PER_HALF_BANK + 1
        k = (hb + 1) * NUM_PLATS_PER_HALF_BANK
        assert all(actual_output[i:k] == 0x0001)
    return out


# +-+-+-+-+-+-+-+-+ +-+
# |E|X|E|R|C|I|S|E| |5|
# +-+-+-+-+-+-+-+-+ +-+


@belex_apl
def exercise_5(Belex, mrk_vreg: VR, mrk_mask: Mask, dst_vreg: VR, val_mask: Mask):
    """Write 1101 1100 0000 0000 from top down in the marked plats of
    SB[3], i.e., write the val_mask in little-endian order. Leave all
    other plats of SB[3] alone.
    """

    # Precondition
    RL[::] <= 0

    #  ___ _              _ _
    # / __| |_ ___ _ __  / (_)
    # \__ \  _/ -_) '_ \ | |_
    # |___/\__\___| .__/ |_(_)
    #             |_|
    #   _____                                __          __         ___  __
    #  / ___/__  ___  __ __  __ _  ___ _____/ /__ ___   / /____    / _ \/ /
    # / /__/ _ \/ _ \/ // / /  ' \/ _ `/ __/  '_/(_-<  / __/ _ \  / , _/ /__
    # \___/\___/ .__/\_, / /_/_/_/\_,_/_/ /_/\_\/___/  \__/\___/ /_/|_/____/
    #         /_/   /___/

    # Step 1: Copy marks to RL:
    #
    # { {  mrk: RL = SB[mrk_vreg];        // instr 3

    RL[mrk_mask] <= mrk_vreg()

    #  ___ _              ___ _
    # / __| |_ ___ _ __  |_  |_)
    # \__ \  _/ -_) '_ \  / / _
    # |___/\__\___| .__/ /___(_)
    #             |_|
    #   _____                                __          __         _______
    #  / ___/__  ___  __ __  __ _  ___ _____/ /__ ___   / /____    / ___/ /
    # / /__/ _ \/ _ \/ // / /  ' \/ _ `/ __/  '_/(_-<  / __/ _ \  / (_ / /__
    # \___/\___/ .__/\_, / /_/_/_/\_,_/_/ /_/\_\/___/  \__/\___/  \___/____/
    #         /_/   /___/

    # Step 2: Copy marks to GL
    #
    #      mrk: GL = RL;  }               // R-sel logic

    GL[mrk_mask] <= RL()

    #  ___ _              _____
    # / __| |_ ___ _ __  |__ (_)
    # \__ \  _/ -_) '_ \  |_ \_
    # |___/\__\___| .__/ |___(_)
    #             |_|
    #   _____     ____        __    _                              _           __
    #  / ___/__  / / /__ ____/ /_  (_)__ _  _____ _______ ___     (_)_ _____  / /__
    # / /__/ _ \/ / / -_) __/ __/ / / _ \ |/ / -_) __(_-</ -_)   / / // / _ \/  '_/
    # \___/\___/_/_/\__/\__/\__/ /_/_//_/___/\__/_/ /___/\__/ __/ /\_,_/_//_/_/\_\
    #                                                        |___/
    #    ___                  __                            ___  _______  ____
    #   / _/_ _    _  _____ _/ / _______ _    _____   ___  / _/ / __/ _ )|_  /
    #  / _/  ' \  | |/ / _ `/ / / __/ _ \ |/|/ (_-<  / _ \/ _/ _\ \/ _  |/_ <
    # /_//_/_/_/  |___/\_,_/_/ /_/  \___/__,__/___/  \___/_/  /___/____/____/

    # Step 3: Read ~S to unmarked plats; zero to marked plats of val rows.
    # VAL-MASK rows (0-1, 3-5),
    # UNMARKED plats (0-2, 4-5, 8+) of SB[3].
    #
    #   {  val: RL = ~SB[dst] & INV_GL;   // instr 8

    # ~diri.GL == 1110 1000 1111 1111 ... (unmarked plats; length 2K)

    RL[val_mask] <= ~dst_vreg() & INV_GL()

    #  ___ _              _ _ _
    # / __| |_ ___ _ __  | | (_)
    # \__ \  _/ -_) '_ \ |_  _|
    # |___/\__\___| .__/   |_(_)
    #             |_|
    #   _____     ____        __      _           __     ___
    #  / ___/__  / / /__ ____/ /_    (_)_ _____  / /__  / _/______  __ _
    # / /__/ _ \/ / / -_) __/ __/   / / // / _ \/  '_/ / _/ __/ _ \/  ' \
    # \___/\___/_/_/\__/\__/\__/ __/ /\_,_/_//_/_/\_\ /_//_/  \___/_/_/_/
    #                           |___/
    #                                 __
    #   ___  ___  ___  _____  _____ _/ / _______ _    _____
    #  / _ \/ _ \/ _ \/___/ |/ / _ `/ / / __/ _ \ |/|/ (_-<
    # /_//_/\___/_//_/    |___/\_,_/_/ /_/  \___/__,__/___/

    # Step 4: Read S to unmarked plats; zeros to marked plats of non-val rows.
    # NON-VAL-MASK (~val) rows (2, 6..F),
    # UNMARKED plats of SB[3]
    #
    #     ~val: RL = SB[dst] & INV_GL;  } // instr 5

    RL[~val_mask] <= dst_vreg() & INV_GL()

    #  ___ _              ___ _
    # / __| |_ ___ _ __  | __(_)
    # \__ \  _/ -_) '_ \ |__ \_
    # |___/\__\___| .__/ |___(_)
    #             |_|
    #  _      __    _ __                  __
    # | | /| / /___(_) /____   _  _____ _/ / _______ _    _____
    # | |/ |/ / __/ / __/ -_) | |/ / _ `/ / / __/ _ \ |/|/ (_-<
    # |__/|__/_/ /_/\__/\__/  |___/\_,_/_/ /_/  \___/__,__/___/

    # Step 5: Write ~~S to unmarked plats, 1 to marked plats of val rows.
    #
    #   {  val: SB[dst] = INV_RL;         // write-logic

    dst_vreg[val_mask] <= INV_RL()

    #  ___ _               __ _
    # / __| |_ ___ _ __   / /(_)
    # \__ \  _/ -_) '_ \ / _ \_
    # |___/\__\___| .__/ \___(_)
    #             |_|
    #  _      __    _ __                                     __
    # | | /| / /___(_) /____   ___  ___  ___  _____  _____ _/ / _______ _    _____
    # | |/ |/ / __/ / __/ -_) / _ \/ _ \/ _ \/___/ |/ / _ `/ / / __/ _ \ |/|/ (_-<
    # |__/|__/_/ /_/\__/\__/ /_//_/\___/_//_/    |___/\_,_/_/ /_/  \___/__,__/___/

    # Step 6: Write S to unmarked plats, 0 to marked plats of non-val rows.
    #
    #     ~val: SB[dst] = RL;  }  };      // write - logic

    dst_vreg[~val_mask] <= RL()


@parameterized_belex_test(repeatably_randomize_half_bank=False)
def test_exercise_5(diri: DIRI) -> int:
    mrk_vreg = 0
    mrk_mask = 0x0004
    dst_vreg = 3
    val_mask = 0x003B

    #    __  ___         __          _______    ___                    ___
    #   /  |/  /__ _____/ /__ ___   / __/ _ )  / _ \    ___ ___ ____  |_  |
    #  / /|_/ / _ `/ __/  '_/(_-<  _\ \/ _  | / // /   (_-</ -_) __/ / __/
    # /_/  /_/\_,_/_/ /_/\_\/___/ /___/____/  \___( ) /___/\__/\__/ /____/
    #                                             |/

    diri.randomize_sb(dst_vreg, seed=44)

    actual = diri.grid(dst_vreg, 7, 10)
    assert actual == \
        'SB[3]  col col col col col col col col col col ... col\n' \
        '______ 000 001 002 003 004 005 006 007 008 009 ... 7FFF\n' \
        'sec 0:  1   1   1   0   1   1   1   0   0   1  ...  0\n' \
        'sec 1:  1   1   0   0   1   0   0   1   1   1  ...  0\n' \
        'sec 2:  0   0   1   1   0   0   0   0   0   0  ...  1\n' \
        'sec 3:  0   1   1   1   1   1   1   0   0   0  ...  1\n' \
        'sec 4:  1   0   0   0   1   1   1   0   0   0  ...  0\n' \
        'sec 5:  1   0   0   1   0   0   0   1   1   0  ...  1\n' \
        'sec 6:  0   0   1   0   1   0   1   0   0   1  ...  0\n' \
        ' ...   ... ... ... ... ... ... ... ... ... ... ... ...\n' \
        'sec F:  0   0   1   0   0   0   1   1   1   0  ...  0'

    # Mark plats 3, 5, 6, 7 by cheating.
    diri.set_marker_cheat(f"0x{mrk_mask:04X}", mrk_vreg, 3)
    diri.set_marker_cheat(f"0x{mrk_mask:04X}", mrk_vreg, 5)
    diri.set_marker_cheat(f"0x{mrk_mask:04X}", mrk_vreg, 6)
    diri.set_marker_cheat(f"0x{mrk_mask:04X}", mrk_vreg, 7)

    actual = diri.grid(mrk_vreg, 7, 10)
    assert actual == \
        'SB[0]  col col col col col col col col col col ... col\n' \
        '______ 000 001 002 003 004 005 006 007 008 009 ... 7FFF\n' \
        'sec 0:  0   0   0   0   0   0   0   0   0   0  ...  0\n' \
        'sec 1:  0   0   0   0   0   0   0   0   0   0  ...  0\n' \
        'sec 2:  0   0   0   1   0   1   1   1   0   0  ...  0\n' \
        'sec 3:  0   0   0   0   0   0   0   0   0   0  ...  0\n' \
        'sec 4:  0   0   0   0   0   0   0   0   0   0  ...  0\n' \
        'sec 5:  0   0   0   0   0   0   0   0   0   0  ...  0\n' \
        'sec 6:  0   0   0   0   0   0   0   0   0   0  ...  0\n' \
        ' ...   ... ... ... ... ... ... ... ... ... ... ... ...\n' \
        'sec F:  0   0   0   0   0   0   0   0   0   0  ...  0'

    exercise_5(mrk_vreg, mrk_mask, dst_vreg, val_mask)

    actual = diri.grid(dst_vreg, 7, 10)
    assert actual == \
        'SB[3]  col col col col col col col col col col ... col\n' \
        '______ 000 001 002 003 004 005 006 007 008 009 ... 7FFF\n' \
        'sec 0:  1   1   1   1   1   1   1   1   0   1  ...  0\n' \
        'sec 1:  1   1   0   1   1   1   1   1   1   1  ...  0\n' \
        'sec 2:  0   0   1   0   0   0   0   0   0   0  ...  1\n' \
        'sec 3:  0   1   1   1   1   1   1   1   0   0  ...  1\n' \
        'sec 4:  1   0   0   1   1   1   1   1   0   0  ...  0\n' \
        'sec 5:  1   0   0   1   0   1   1   1   1   0  ...  1\n' \
        'sec 6:  0   0   1   0   1   0   0   0   0   1  ...  0\n' \
        ' ...   ... ... ... ... ... ... ... ... ... ... ... ...\n' \
        'sec F:  0   0   1   0   0   0   0   0   1   0  ...  0'

    return dst_vreg


# +-+-+-+-+-+-+-+-+ +-+
# |E|X|E|R|C|I|S|E| |6|
# +-+-+-+-+-+-+-+-+ +-+


@belex_apl
def set_sec_0_plat_0(Belex, mvr: VR):

    # new general case for laning: when sections overlap
    # but unconditionally overwrite in the overlap sections

    # Not lanable
    # RL["0xFFFF"] <= 0  # This section mask will change ...
    # RL[0] <= 1
    #     # RL now has 1111 1111 1111 ... in sec 0

    # Lanable (and laned)
    with apl_commands():
        RL["0xFFFE"] <= 0  # ... to the XOR of FFFF and 0001
        RL[0] <= 1
        # RL now has 1111 1111 1111 ... in sec 0

    # Can't lane this with the next; only 'read-before-broadcast'
    # is lanable. These two would be broadcast-before-read. Try
    # laning them and watch the test fail.

    mvr[:] <= WRL()
    # mvr now has 0111 1111 1111 ... in sec 0

    # Can't lane this with the next; only 'write-before-read' is
    # lanable. These two would be read-before-write. Try laning
    # them and watch the test fail.

    RL[0] <= mvr()

    mvr[0] <= ~RL()
    # mvr now has 1000 0000 0000 ... in sec 0


@belex_apl
def clear_vr(Belex, vr: VR):
    # vr[:] <= RSP16()
    RL[:] <= 0
    vr[:] <= RL()


@belex_apl
def exercise_6(Belex,
               mvr: VR,      index_vr: VR,
               match_vr: VR, dest_vr: VR,
               mrk_sec: Section):
    """Mark 5 plats, each shifted one more than the last.
    - Write the numbers 1-5 in the first 5 plats
    - Write test numbers into those plats in another VR
    """

    # These two possibilities for clearing a VR don't work
    # at present due to interactions with GVML under the
    # covers.
    # cpy_imm_16(index_vr, "0x0000")  # doesn't work now
    # set_16(index_vr)  # doesn't do what you think it does

    clear_vr(index_vr)

    set_sec_0_plat_0(mvr)

    write_to_marked(index_vr, mvr, mrk_sec, "0x0001")

    walk_marks_eastward(mvr, mrk_sec)
    write_to_marked(index_vr, mvr, mrk_sec, "0x0002")

    walk_marks_eastward(mvr, mrk_sec)
    write_to_marked(index_vr, mvr, mrk_sec, "0x0003")

    walk_marks_eastward(mvr, mrk_sec)
    write_to_marked(index_vr, mvr, mrk_sec, "0x0004")

    walk_marks_eastward(mvr, mrk_sec)
    write_to_marked(index_vr, mvr, mrk_sec, "0x0005")

    clear_vr(match_vr)
    write_markers_in_plats_matching_value(
        index_vr,  # search this VR
        "0x0003",  # for this value (as a u16)
        match_vr,  # and put a marker in this VR
        mrk_sec    # and this section
    )
    clear_vr(dest_vr)
    write_to_marked(dest_vr, match_vr, mrk_sec, "0x000D")  # arbitrary


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_exercise_6(diri: DIRI) -> int:

    r""""""

    mrk_vr   = 0
    index_vr = 1
    match_vr = 2
    dst_vr   = 3

    mrk_sec  = 0

    exercise_6(mrk_vr, index_vr, match_vr, dst_vr, mrk_sec)

    actual_mrk_vr_ = diri.glass(mrk_vr, sections=4, plats=12)
    actual_mrk_vr = ''.join(actual_mrk_vr_.split('\n'))
    assert actual_mrk_vr == '0000''1000''0000' \
                            '0000''0000''0000' \
                            '0000''0000''0000' \
                            '0000''0000''0000'

    actual_index_ = diri.glass(index_vr, sections=4, plats=12)
    actual_index = ''.join(actual_index_.split('\n'))
    assert actual_index == '1010''1000''0000' \
                           '0110''0000''0000' \
                           '0001''1000''0000' \
                           '0000''0000''0000'

    actual_match_ = diri.glass(match_vr, sections=4, plats=12)
    actual_match = ''.join(actual_match_.split('\n'))
    assert actual_match == '0010''0000''0000' \
                           '0000''0000''0000' \
                           '0000''0000''0000' \
                           '0000''0000''0000'

    actual_dst_ = diri.glass(dst_vr, sections=4, plats=12)
    actual_dst = ''.join(actual_dst_.split('\n'))
    assert actual_dst == '0010''0000''0000' \
                         '0000''0000''0000' \
                         '0010''0000''0000' \
                         '0010''0000''0000'

    return mrk_vr



# +-+-+-+-+-+-+-+-+ +-+-+ +-+-+-+-+-+-+-+-+-+
# |E|X|E|R|C|I|S|E| |7|:| |I|N|C|R|E|M|E|N|T|
# +-+-+-+-+-+-+-+-+ +-+-+ +-+-+-+-+-+-+-+-+-+


# This can and should be improved using the GVML adder.
# However, it has value as a pedagogical exercise on
# writing a 'brute-force' adder and as a regression
# test /cum/ unit test.


@belex_apl
def brute_ripple_carry(Belex, sm: VR, cy: VR):
    sm[:] = RL()

    with apl_commands("read before b'cast"):
        RL[:] = 1
        GL[:] = RL()

    RL[ 0]  <= sm() & GL()
    cy[ 1]  <= NRL()
    RL[ 0]  <= sm() ^ GL()
    sm[ 0]  <= RL()

    RL[ 1]  <= cy()
    RL[ 1]  <= sm() & RL()
    cy[ 2]  <= NRL()
    RL[ 1]  <= cy()
    RL[ 1]  <= sm() ^ RL()
    sm[ 1]  <= RL()

    RL[ 2]  <= cy()
    RL[ 2]  <= sm() & RL()
    cy[ 3]  <= NRL()
    RL[ 2]  <= cy()
    RL[ 2]  <= sm() ^ RL()
    sm[ 2]  <= RL()

    RL[ 3]  <= cy()
    RL[ 3]  <= sm() & RL()
    cy[ 4]  <= NRL()
    RL[ 3]  <= cy()
    RL[ 3]  <= sm() ^ RL()
    sm[ 3]  <= RL()

    RL[ 4]  <= cy()
    RL[ 4]  <= sm() & RL()
    cy[ 5]  <= NRL()
    RL[ 4]  <= cy()
    RL[ 4]  <= sm() ^ RL()
    sm[ 4]  <= RL()

    RL[ 5]  <= cy()
    RL[ 5]  <= sm() & RL()
    cy[ 6]  <= NRL()
    RL[ 5]  <= cy()
    RL[ 5]  <= sm() ^ RL()
    sm[ 5]  <= RL()

    RL[ 6]  <= cy()
    RL[ 6]  <= sm() & RL()
    cy[ 7]  <= NRL()
    RL[ 6]  <= cy()
    RL[ 6]  <= sm() ^ RL()
    sm[ 6]  <= RL()

    RL[ 7]  <= cy()
    RL[ 7]  <= sm() & RL()
    cy[ 8]  <= NRL()
    RL[ 7]  <= cy()
    RL[ 7]  <= sm() ^ RL()
    sm[ 7]  <= RL()

    RL[ 8]  <= cy()
    RL[ 8]  <= sm() & RL()
    cy[ 9]  <= NRL()
    RL[ 8]  <= cy()
    RL[ 8]  <= sm() ^ RL()
    sm[ 8]  <= RL()

    RL[ 9]  <= cy()
    RL[ 9]  <= sm() & RL()
    cy[10]  <= NRL()
    RL[ 9]  <= cy()
    RL[ 9]  <= sm() ^ RL()
    sm[ 9]  <= RL()

    RL[10]  <= cy()
    RL[10]  <= sm() & RL()
    cy[11]  <= NRL()
    RL[10]  <= cy()
    RL[10]  <= sm() ^ RL()
    sm[10]  <= RL()

    RL[11]  <= cy()
    RL[11]  <= sm() & RL()
    cy[12]  <= NRL()
    RL[11]  <= cy()
    RL[11]  <= sm() ^ RL()
    sm[11]  <= RL()

    RL[12]  <= cy()
    RL[12]  <= sm() & RL()
    cy[13]  <= NRL()
    RL[12]  <= cy()
    RL[12]  <= sm() ^ RL()
    sm[12]  <= RL()

    RL[13]  <= cy()
    RL[13]  <= sm() & RL()
    cy[14]  <= NRL()
    RL[13]  <= cy()
    RL[13]  <= sm() ^ RL()
    sm[13]  <= RL()

    RL[14]  <= cy()
    RL[14]  <= sm() & RL()
    cy[15]  <= NRL()
    RL[14]  <= cy()
    RL[14]  <= sm() ^ RL()
    sm[14]  <= RL()

    RL[15]  <= cy()
    RL[15]  <= sm() & RL()
    RL[15]  <= cy()
    RL[15]  <= sm() ^ RL()
    sm[15]  <= RL()


@belex_apl
def exercise_7(Belex, sm: VR):

    """1. Implement increment. Input a value, say 7
          across all plats in all HBs

         - Assume multi-section as in all prior examples

         - Add 1 to the LSB section using truth table below

           |-------+-------+-------+-------|
           |       |       |  Sum  | Carry |
           |   A   |   B   | A ^ B | A & B |
           | SB[0] |   1   |       |       |
           |-------+-------+-------+-------|
           |   0   |   0   |   0   |   0   |
           |   0   |   1   |   1   |   0   |
           |   1   |   0   |   1   |   0   |
           |   1   |   1   |   0   |   1   |
           |-------+-------+-------+-------|

         - Shift "carry" south to next section.

         - Repeat the truth-table operation"""

    cy = Belex.VR()  # select a temporary VR for carries

    cpy_imm_16_to_rl("0x0007")
    brute_ripple_carry(sm, cy)

    return sm


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_exercise_7(diri: DIRI) -> int:

    r""""""

    sb0 = 0

    exercise_7(sb0)

    corner_view = diri.glass(sb=0, sections=16, plats=8)
    assert corner_view == '00000000\n' \
                          '00000000\n' \
                          '00000000\n' \
                          '11111111\n' \
                          \
                          '00000000\n' \
                          '00000000\n' \
                          '00000000\n' \
                          '00000000\n' \
                          \
                          '00000000\n' \
                          '00000000\n' \
                          '00000000\n' \
                          '00000000\n' \
                          \
                          '00000000\n' \
                          '00000000\n' \
                          '00000000\n' \
                          '00000000'

    actual = diri.ndarray_1d_to_bitstring(diri.GL[0:40])
    assert actual == '1111111111''1111111111''1111111111''1111111111'

    return sb0


# +-+-+-+-+-+-+-+-+ +-+-+ +-+-+-+-+-+-+-+ +-+-+-+-+-+
# |E|X|E|R|C|I|S|E| |8|:| |r|u|n|n|i|n|g| |i|n|d|e|x|
# +-+-+-+-+-+-+-+-+ +-+-+ +-+-+-+-+-+-+-+ +-+-+-+-+-+

# @pretty_print
@belex_apl
def walk_right(Belex, sb: VR):
    sb[:] <= WRL()
    RL[:] <= sb()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_exercise_8(diri: DIRI) -> int:
    sm = 0
    cy = 8  # select a temporary VR for carries

    plats = 32
    clear_vr(sm)
    for _ in range(0, plats):
        brute_ripple_carry(sm, cy)
        if _ != (plats - 1):
            walk_right(sm)

    sm_glass = diri.glass(sm, plats=32).split('\n')
    assert sm_glass == \
    ['1010''1010''1010''1010''1010''1010''1010''1010',
     '0110''0110''0110''0110''0110''0110''0110''0110',
     '0001''1110''0001''1110''0001''1110''0001''1110',
     '0000''0001''1111''1110''0000''0001''1111''1110',
     '0000''0000''0000''0001''1111''1111''1111''1110',
     '0000''0000''0000''0000''0000''0000''0000''0001',
     '0000''0000''0000''0000''0000''0000''0000''0000',
     '0000''0000''0000''0000''0000''0000''0000''0000',
     '0000''0000''0000''0000''0000''0000''0000''0000',
     '0000''0000''0000''0000''0000''0000''0000''0000',
     '0000''0000''0000''0000''0000''0000''0000''0000',
     '0000''0000''0000''0000''0000''0000''0000''0000',
     '0000''0000''0000''0000''0000''0000''0000''0000',
     '0000''0000''0000''0000''0000''0000''0000''0000',
     '0000''0000''0000''0000''0000''0000''0000''0000',
     '0000''0000''0000''0000''0000''0000''0000''0000']

    return sm


#  ___          _    _
# | _ \___ __ _(_)__| |_ ___ _ _
# |   / -_) _` | (_-<  _/ -_) '_|
# |_|_\___\__, |_/__/\__\___|_|
#         |___/
#  __  __                                       _
# |  \/  |__ _ _ _  __ _ __ _ ___ _ __  ___ _ _| |_
# | |\/| / _` | ' \/ _` / _` / -_) '  \/ -_) ' \  _|
# |_|  |_\__,_|_||_\__,_\__, \___|_|_|_\___|_||_\__|
#                       |___/


@belex_apl
def exercise_6_too_many_vrs_for_rn_regs(
        Belex,
        # VR parameters; too many for our miniaturized
        # machine, which has only three RN_REGs.
        mvr: VR,          index_vr: VR,
        match_vr: VR,     dest_vr: VR,
        # section and mask parameters
        mrk_sec: Section, msk_1: Mask,
        msk_2: Mask,      msk_3: Mask,
        msk_4: Mask,      msk_5: Mask):

    """Mark 5 plats, each shifted one more than the last.
    - Write the numbers 1-5 in the first 5 plats
    - Write test numbers into those plats in another VR
    """

    # The dataflow analysis for liveness is
    # straightforward, here, because there is no branching
    # and no calls. All loops are unrolled and all calls
    # are inlined.

    # Liveness analysis maintains FOUR mathematical sets
    # (no duplicate elements, no order of elements) at
    # each instruction: GEN, KILL, IN, and OUT.
    #
    # GEN  :: all VRs read (RL = ...) in an instruction
    #
    # KILL :: all VRs written (SB = ...) in an instruction
    #
    #         GEN and KILL may overlap
    #
    # IN   :: all VRs live prior to the instruction =
    #         OUT - KILL + GEN
    #
    # OUT  :: all VRs live after the instruction =
    #         union(IN) over all successor instructions.

    # Let us write sets in a lightweight notation:
    # OUT = d+h+x+m means OUT = {d} \/ {h} \/ {x} \/ {m}
    # = {d, h, x, m} if d, h, x, m are distinct, and
    # where {x} is a singleton set containing x and \/ is
    # set union.

    # For IN and OUT, Start with the last instruction and
    # work backwards. OUT at the end is the returned VR.
    # Let's do this calculation by hand on the generated
    # APL (could just as well do it on the BLEIR). The
    # following is pseudocode. APL takes in RN_REGs, but
    # let's pretend that m, x, h, and d are VRs and that
    # we're going to assign them to RN_REGs named X, Y, Z.
    # Ignore SM_REGs for this analysis, but we can handle
    # them in a similar way.

    # -------------------------------------------------------------
    # BEGINNING OF REGISTER-ALLOCATION HAND CALCULATION
    # -------------------------------------------------------------
    # /**
    #  * Original nym: exercise_6_too_many_vrs_for_rn_regs
    #  */
    # APL_FRAG exercise_6_to_7bf0bb25fc(
    #         RN_REG m,  # marker_vr
    #         RN_REG x,  # index_vr
    #         RN_REG h,  # match_vr
    #         RN_REG d,  # dest_vr
    #         SM_REG mrk_sec,
    #         SM_REG msk_1,
    #         SM_REG msk_2,
    #         SM_REG msk_3,
    #         SM_REG msk_4,
    #         SM_REG msk_5,
    #         SM_REG _INTERNAL_SM_0X0003,
    #         SM_REG _INTERNAL_SM_0X000D)
    # {   SM_0XFFFF: RL = 0;                       # GEN = KILL = (/); OUT = d+h+x+m; IN = d+h+x+m
    #
    #     # For the next instruction (an unlaned, single
    #     # statement), notice that KILL = x, so we must
    #     # ensure that's in an RN_REG. Scanning forward,
    #     # the next GEN is m, so it's a good idea to keep
    #     # that in an RN_REG, too, at this point. So
    #     # we're going to spill either h or d. The next
    #     # GEN after m is h and we're now out of RN_REGs,
    #     # so let's leave d spilled and restore the
    #     # others. Use the same function to RESTORE as to
    #     # initialize. We won't need to spill m (from Y)
    #     # and restore d (to Y) for a while, so the hand
    #     # calculation will be a bit tedious.
    #
    #     RESTORE VR x from named memory to RN_REG X;
    #     RESTORE VR m from named memory to RN_REG Y;
    #     RESTORE VR h from named memory tp RN_REG Z;
    #
    #     SM_0XFFFF: SB[X=x] = RL;                 # GEN = (/); KILL = x; OUT = d+h+x+m; IN = d+h+m
    #     {                                        # GEN = KILL = (/); OUT = d+h+x+m; IN = d+h+x+m
    #     SM_0XFFFF<<1: RL = 0;
    #     SM_0X0001: RL = 1;                       #   from this point on,
    #     }                                        #   (/), empty, is default
    #     SM_0XFFFF: SB[Y=m] = WRL;                # GEN = m; KILL = m; OUT = d+h+x+m; IN = d+h+x
    #     SM_0X0001: RL = SB[Y=m];
    #     SM_0X0001: SB[Y=m] = INV_RL;
    #     SM_0XFFFF: RL = 0;
    #     /* Copy marks to GL. */
    #     {                                        # GEN = m; OUT = d+h+x+m; IN = d+h+x+m
    #     mrk_sec: RL = SB[Y=m];
    #     mrk_sec: GL = RL;
    #     }
    #     /* Copy original data to unmarked plats. */
    #     {                                        # GEN = x; OUT = d+h+x+m; IN = d+h+x+m
    #     msk_1: RL = ~SB[X=x] & INV_GL;
    #     ~msk_1: RL = SB[X=x] & INV_GL;
    #     }
    #     /* Copy back to marked plats. */
    #     {                                        # KILL = x; OUT = d+h+x+m; IN = d+h+m
    #     msk_1: SB[X=x] = INV_RL;
    #     ~msk_1: SB[X=x] = RL;
    #     }
    #     mrk_sec: RL = SB[Y=m];                   # GEN = m; OUT = d+h+x+m; IN = d+h+x+m
    #     mrk_sec: SB[Y=m] = WRL;                  # KILL = m; OUT = d+h+x+m; IN = d+h+x
    #     SM_0XFFFF: RL = 0;
    #     /* Copy marks to GL. */
    #     {                                        # GEN = m; OUT = d+h+x+m; IN = d+h+x+m
    #     mrk_sec: RL = SB[Y=m];
    #     mrk_sec: GL = RL;
    #     }
    #     /* Copy original data to unmarked plats. */
    #     {                                        # GEN = x; OUT = d+h+x+m; IN = d+h+x+m
    #     msk_2: RL = ~SB[X=x] & INV_GL;
    #     ~msk_2: RL = SB[X=x] & INV_GL;
    #     }
    #     /* Copy back to marked plats. */
    #     {                                        # KILL = x; OUT = d+h+x+m; IN = d+h+m
    #     msk_2: SB[X=x] = INV_RL;
    #     ~msk_2: SB[X=x] = RL;
    #     }
    #     mrk_sec: RL = SB[Y=m];                   # GEN = m; OUT = d+h+x+m; IN = d+h+x+m
    #     mrk_sec: SB[Y=m] = WRL;                  # KILL = m; OUT = d+h+x+m; IN = d+h+x
    #     SM_0XFFFF: RL = 0;                       # GEN = KILL = (/)
    #     /* Copy marks to GL. */
    #     {                                        # GEN = m; OUT = d+h+x+m; IN = d+h+x+m
    #     mrk_sec: RL = SB[Y=m];
    #     mrk_sec: GL = RL;
    #     }
    #     /* Copy original data to unmarked plats. */
    #     {                                        # GEN = x; OUT = d+h+x+m; IN = d+h+x+m
    #     msk_3: RL = ~SB[X=x] & INV_GL;
    #     ~msk_3: RL = SB[X=x] & INV_GL;
    #     }
    #     /* Copy back to marked plats. */
    #     {                                        # KILL = x; OUT = d+h+x+m; IN = d+h++m
    #     msk_3: SB[X=x] = INV_RL;
    #     ~msk_3: SB[X=x] = RL;
    #     }
    #     mrk_sec: RL = SB[Y=m];                   # GEN = m; OUT = d+h+x+m; IN = d+h+x+m
    #     mrk_sec: SB[Y=m] = WRL;                  # KILL = m; OUT = d+h+x+m; IN = d+h+x
    #     SM_0XFFFF: RL = 0;                       # GEN = KILL = (/); OUT = d+h+x+m; IN = d+h+x+m
    #     /* Copy marks to GL. */
    #     {                                        # GEN = m; OUT = d+h+x+m; IN = d+h+x+m
    #     mrk_sec: RL = SB[Y=m];
    #     mrk_sec: GL = RL;
    #     }
    #     /* Copy original data to unmarked plats. */
    #     {                                        # GEN = x; OUT = d+h+x+m; IN = d+h+x+m
    #     msk_4: RL = ~SB[X=x] & INV_GL;
    #     ~msk_4: RL = SB[X=x] & INV_GL;
    #     }
    #     /* Copy back to marked plats. */
    #     {                                        # KILL = x; OUT = d+h+x+m; IN = d+h+m
    #     msk_4: SB[X=x] = INV_RL;
    #     ~msk_4: SB[X=x] = RL;
    #     }
    #     mrk_sec: RL = SB[Y=m];                   # GEN = m; OUT = d+h+x+m; IN = d+h+x+m
    #     mrk_sec: SB[Y=m] = WRL;                  # KILL = m; OUT = d+h+x+m; IN = d+h+x
    #     SM_0XFFFF: RL = 0;                       # GEN = KILL = (/); OUT = d+h+x+m; IN = d+h+x+m
    #     /* Copy marks to GL. */
    #     {                                        # GEN = m; OUT = d + h + x; IN = d + h + x + m
    #     mrk_sec: RL = SB[Y=m];
    #     mrk_sec: GL = RL;
    #     }
    #     /* Copy original data to unmarked plats. */
    #     {                                        # GEN = x; OUT = d + h + x; IN = d + h + x
    #     msk_5: RL = ~SB[X=x] & INV_GL;
    #     ~msk_5: RL = SB[X=x] & INV_GL;
    #     }
    #     /* Copy back to marked plats. */
    #     {                                        # KILL = x; OUT = d + h + x; IN = d + h
    #     msk_5: SB[X=x] = INV_RL;
    #     ~msk_5: SB[X=x] = RL;
    #     }
    #     SM_0XFFFF: RL = 0;                       # GEN = KILL = (/)
    #     SM_0XFFFF: SB[Z=h] = RL;                 # KILL = h; OUT = d + h + x; IN = d + x
    #     {
    #     ~_INTERNAL_SM_0X0003: RL = 0;
    #     _INTERNAL_SM_0X0003: RL = 1;
    #     }
    #     SM_0XFFFF: RL = SB[X=x] ^ INV_RL;        # GEN = x; OUT = d + h; IN = d + h + x
    #     SM_0XFFFF: GL = RL;                      # GEN = KILL = (/); OUT = d + h = IN
    #     mrk_sec: SB[Z=h] = GL;                   # KILL = h; OUT = d + h; IN = d
    #     SM_0XFFFF: RL = 0;                       # GEN = KILL = (/); OUT = d + h = IN
    #
    #     # Here, at last, is the fateful moment. We must
    #     # spill m from Y and restore d to Y. We won't
    #     # need to reverse it till the end because m is
    #     # no longer in any IN set.
    #
    #     SPILL RN_REG Y to named memory for VR m;
    #     RESTORE VR d from named memory tp RN_REG Y;
    #
    #     SM_0XFFFF: SB[Y=d] = RL;                 # KILL = d; OUT = d + h; IN = h
    #     SM_0XFFFF: RL = 0;                       # GEN = KILL = (/); OUT = d + h = IN
    #     /* Copy marks to GL. */
    #     {                                        # GEN = h; OUT = d; IN = d + h
    #     mrk_sec: RL = SB[Z=h];
    #     mrk_sec: GL = RL;
    #     }
    #     /* Copy original data to unmarked plats. */
    #     {                                        # GEN = d; OUT = (/); IN = d
    #     _INTERNAL_SM_0X000D: RL = ~SB[Y=d] & INV_GL;  # OUT = (/)
    #     ~_INTERNAL_SM_0X000D: RL = SB[Y=d] & INV_GL;  # IN = d
    #     }
    #     /* Copy back to marked plats. */
    #     {                                        # KILL = d; OUT = d; IN = (/)
    #     _INTERNAL_SM_0X000D: SB[Y=d] = INV_RL;
    #     ~_INTERNAL_SM_0X000D: SB[Y=d] = RL;
    #     }
    #     SPILL RN_REG Y to named memory for VR d;
    #     };
    # -------------------------------------------------------------
    # END OF REGISTER-ALLOCATION HAND CALCULATION
    # -------------------------------------------------------------


    # These two possibilities for clearing a VR don't work
    # at present due to interactions with GVML under the
    # covers.

    # cpy_imm_16(index_vr, "0x0000")  # doesn't work now
    # set_16(index_vr)  # doesn't do what you think it does

    # We wrote a little routine in this file to do our
    # clearing for us.
    clear_vr(index_vr)

    # Mark sec 0, plat 0 in mvr.
    set_sec_0_plat_0(mvr)

    # write a number, represented by a mask, msk_1, to
    # index_vr
    write_to_marked(index_vr, mvr, mrk_sec, msk_1)

    walk_marks_eastward(mvr, mrk_sec)
    write_to_marked(index_vr, mvr, mrk_sec, msk_2)

    walk_marks_eastward(mvr, mrk_sec)
    write_to_marked(index_vr, mvr, mrk_sec, msk_3)

    walk_marks_eastward(mvr, mrk_sec)
    write_to_marked(index_vr, mvr, mrk_sec, msk_4)

    walk_marks_eastward(mvr, mrk_sec)
    write_to_marked(index_vr, mvr, mrk_sec, msk_5)

    clear_vr(match_vr)
    write_markers_in_plats_matching_value(
        index_vr,  # search this VR
        "0x0003",  # for this value (as a u16)
        match_vr,  # and put a marker in this VR
        mrk_sec    # and this section
    )
    clear_vr(dest_vr)
    write_to_marked(dest_vr, match_vr, mrk_sec, "0x000D")  # arbitrary


def check_glass(diri: DIRI, vr_num, expected_value, sections=4, plats=12):
    actual__ = diri.glass(vr_num, sections=sections, plats=plats)
    actual_ = ''.join(actual__.split('\n'))
    assert actual_ == expected_value


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_exercise_6_too_many_vrs_for_rn_regs(diri: DIRI) -> int:

    r""""""

    mrk_vr   = 0
    index_vr = 1
    match_vr = 2
    dst_vr   = 3

    mrk_sec  = 0

    msk_1 = 1
    msk_2 = 2
    msk_3 = 3
    msk_4 = 4
    msk_5 = 5

    exercise_6_too_many_vrs_for_rn_regs(
        mrk_vr, index_vr, match_vr, dst_vr,
        mrk_sec, msk_1, msk_2, msk_3, msk_4, msk_5)

    # Check that the marker register, after the last walk,
    # has the marker in plat 4, the fifth plat.
    check_glass(diri, mrk_vr,   '0000''1000''0000'
                                '0000''0000''0000'
                                '0000''0000''0000'
                                '0000''0000''0000')

    # Check that the index contains consecutive integers
    # in little-endian from top sections on down.
    check_glass(diri, index_vr, '1010''1000''0000'
                                '0110''0000''0000'
                                '0001''1000''0000'
                                '0000''0000''0000')

    # Match '3' in index_vr's plat number 3. Mark that
    # plat in 'match_vr'.
    check_glass(diri, match_vr, '0010''0000''0000'
                                '0000''0000''0000'
                                '0000''0000''0000'
                                '0000''0000''0000')

    # Check that 0xD (decimal 13, binary 8 + 4 + 1) has
    # been written to dst_vr.
    check_glass(diri, dst_vr,   '0010''0000''0000'
                                '0000''0000''0000'
                                '0010''0000''0000'
                                '0010''0000''0000')

    return mrk_vr

    # =============================== #
    # Case of more variables than VRs #
    # =============================== #

    # I think the same algorithm that spills and restores
    # VRs to RN_REGs might work for spilling and restoring
    # variables to VRs. Variables to VRs works at the
    # level of frag-caller calls, where variables are
    # assigned actual VR numbers. VRs to RN_REGs works at
    # the level of frag-callers, where RN_REGS are
    # assigned actual VR numbers.

    # If there are more variables alive at a given point
    # than there are VRs to assign to them, then we must
    # spill and restore the variables. Let's reason in
    # miniature. Imagine there are only four VRs, X, Y, Z,
    # W, but there are five variables, a, b, c, d, e. The
    # spill storage is named storage: it's named after the
    # variables. Start with VR X's containing variable a's
    # data, VR Y's containing b's data, VR Z's containing
    # c's data, and VR W's containing d's data. When the
    # code need the data in variable e, do this:

    # 1. Spill W's contents to named storage for d.
    # 2. Restore named storage for e to W.

    # and vice versa when d' data is needed. The compiler
    # must remember, at each point, whether d or e is
    # needed. If both are needed, the compiler must spill
    # and restore more than one VR.

    # After an initial pass, the compiler has a list of
    # variables live at every point in the code. A live
    # variable

    # ============================= #
    # Case of more VRs than RN_REGs #
    # ============================= #

    # Reasoning in miniature: say there are four VRs and
    # only three RN_REGs.

    # VR a, b, c, d;  // VRs (in the role of variables)
    # VR *x, *y, *z;  // Model RN_REG as a pointer to a VR

    # // Invariant in our analogy is we want to assign the
    # // pointer values once, before calling APL. APL can
    # // change the contents of a, b, c, and d, but it
    # // can't change the values of x, y, z.

    # // BEFORE CALLING APL

    # x = & a;
    # y = & b;
    # z = & c;

    # // Brute force: spill all.

    # tc = *z;  // spill *z to named storage for c
    # td =  d;  // spill  d to named storage for d

    # // Refinement: spill only the first one that will be
    # // needed later (generalizes to more than 2
    # // variables).

    # if (c is used first)
    #     td = d;  // First one we musts put into c
    # else
    #     tc = c;

    # // PRECONDITION z points to the value of c, *z == c
    # // because compiler determines that c is live first.

    # // INSIDE APL

    # *z == c // use this for a while

    # // Now we need *z == d, but always z == & c
    # (invariant);

    # // Swap d for c in z
    # tc = *z;  // spill   *z to   named storage for c
    # *z = td;  // restore *z from named storage for d

    # // Now we need again *z == c

    # // Swap c for d in z
    # td = *z;  // spill   *z to   named storage for d
    # *z = tc;  // restore *z from named stroage for c

    # // Now we need again *z == d

    # // Swap d for c in z
    # tc = *z;  // spill   *z to   named storage for c
    # *z = td;  // restore *z from named storage for d

    # // ON EXITING THE APL

    # // The reason the POSTPROCESSING is asymmetric is
    # // because z began life pointing to c.

    # case 1: *z contains c
    # case 2: *z contains d
    #     spill *z to named storage for d
    #     restore c from named storage for c
    # restore d from named storage for d


#  _  _      _ _   _____       _
# | \| |_  _| | | |_   _|__ __| |_ ___
# | .` | || | | |   | |/ -_|_-<  _(_-<
# |_|\_|\_,_|_|_|   |_|\___/__/\__/__/


# We want some minimal tests to exercise the code
# generator as if BELEX were an AOT (Ahead-Of-Time)
# compiler.

# Suppose you wrote a @belex_apl like

#   exercise_6_too_many_vrs_for_rn_regs(
#     mrk_vr, index_vr, match_vr, dst_vr,
#     mrk_sec, msk_1, msk_2, msk_3, msk_4, msk_5)

# We could automatically generate a dummy test for it like
# this:

@parameterized_belex_test
def test_dummy_exercise_6_too_many_vrs(diri: DIRI) -> int:

    mrk_vr   = 0
    index_vr = 1
    match_vr = 2
    dst_vr   = 3

    mrk_sec  = 0

    msk_1 = 1
    msk_2 = 2
    msk_3 = 3
    msk_4 = 4
    msk_5 = 5

    exercise_6_too_many_vrs_for_rn_regs(
        mrk_vr, index_vr, match_vr, dst_vr,
        mrk_sec, msk_1, msk_2, msk_3, msk_4, msk_5)

    return mrk_vr

# Don't even need to run the test, just fish the compiled
# APL and .h files from the artifact directory.

# I'll go do something similar for high-level BELEX.
