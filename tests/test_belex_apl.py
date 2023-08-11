r"""By Dylon Edwards

Pytest-only tests of BELEX code generation.

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

import pytest

import open_belex.bleir.types as BLEIR
from open_belex.apl import (APL_comment, APL_masked_stmt, APL_rl_from_sb,
                            APL_rl_from_sb_src, APL_rl_from_src,
                            APL_sb_from_src, APL_set_rl, Mask,
                            collect_all_normalized_masks_from_masked_stmt,
                            make_bleir_map)
from open_belex.bleir.syntactic_validators import validate_types


def to_bit_indices(mask):
    indices = []
    for i in range(16):
        if mask & 0x1 == 1:
            indices.append(i)
        mask = mask >> 1
    return indices


def test_to_bit_indices():
    mask = 0x0000
    assert [] == to_bit_indices(mask)

    mask = 0x0101
    assert [0, 8] == to_bit_indices(mask)

    mask = 0xC10F
    assert [0, 1, 2, 3, 8, 14, 15] == to_bit_indices(mask)


def test_mask_init():
    sm = 0x0000
    msk = Mask([])
    assert sm == msk.mask

    sm = 0x0101
    msk = Mask([0, 8])
    assert sm == msk.mask

    sm = 0xC10F
    msk = Mask([0, 1, 2, 3, 8, 14, 15])
    assert sm == msk.mask


def test_mask_get_shift():
    with pytest.raises(RuntimeError):
        # Mask is 0x0000 so there is no shift
        msk = Mask([])
        msk.get_shift()

    msk = Mask([0])
    assert 0 == msk.get_shift()

    msk = Mask([1])
    assert 1 == msk.get_shift()

    msk = Mask([8])
    assert 8 == msk.get_shift()

    msk = Mask([15])
    assert 15 == msk.get_shift()

    msk = Mask([2, 3, 5, 6, 7])
    assert 2 == msk.get_shift()


def test_mask_get_normalized():
    mask = Mask(to_bit_indices(0xFFFF))
    assert ('0xffff', 0) == mask.get_normalized()

    mask = Mask(to_bit_indices(0xFFF0))
    assert ('0xffff', 4) == mask.get_normalized()

    mask = Mask(to_bit_indices(0xFF00))
    assert ('0xffff', 8) == mask.get_normalized()

    mask = Mask(to_bit_indices(0xF000))
    assert ('0xffff', 12) == mask.get_normalized()

    with pytest.raises(RuntimeError):
        mask = Mask(to_bit_indices(0x0000))
        mask.get_normalized()

    mask = Mask(to_bit_indices(0x0101))
    assert ('0x0101', 0) == mask.get_normalized()

    mask = Mask(to_bit_indices(0x1010))
    assert ('0x0101', 4) == mask.get_normalized()

    mask = Mask(to_bit_indices(0x0001))
    assert ('0x0001', 0) == mask.get_normalized()

    mask = Mask(to_bit_indices(0x0010))
    assert ('0x0001', 4) == mask.get_normalized()

    mask = Mask(to_bit_indices(0x0100))
    assert ('0x0001', 8) == mask.get_normalized()

    mask = Mask(to_bit_indices(0x1000))
    assert ('0x0001', 12) == mask.get_normalized()


def test_apl_comment_render_bleir():
    message = "This is a comment"
    comment = APL_comment(message)
    rn_registers = {}
    bleir_map = make_bleir_map()
    expected_value = BLEIR.SingleLineComment(message)
    actual_value = comment.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)


def test_apl_comment_str():
    message = "This is a comment"
    comment = APL_comment(message)
    assert str(comment) == "/* This is a comment */"


def test_apl_sb_from_src_render_bleir():
    src = "RL"

    # One RN_REG
    # ----------

    sbs = [1]
    sb_from_src = APL_sb_from_src(sbs, src)

    rn_reg_t1 = BLEIR.RN_REG(f"rn_reg_t1")
    rn_registers = {1: rn_reg_t1.identifier}

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1

    expected_value = BLEIR.assign(BLEIR.SB[rn_reg_t1], BLEIR.RL)
    actual_value = sb_from_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Two RN_REGs
    # -----------

    sbs = [3, 9]
    sb_from_src = APL_sb_from_src(sbs, src)

    rn_reg_t3 = BLEIR.RN_REG(f"rn_reg_t3")
    rn_reg_t9 = BLEIR.RN_REG(f"rn_reg_t9")
    rn_registers = {
        3: rn_reg_t3.identifier,
        9: rn_reg_t9.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t3.identifier] = rn_reg_t3
    bleir_map[rn_reg_t9.identifier] = rn_reg_t9

    expected_value = BLEIR.assign(BLEIR.SB[rn_reg_t3, rn_reg_t9], BLEIR.RL)
    actual_value = sb_from_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Three RN_REGs
    # -------------

    sbs = [2, 5, 7]
    sb_from_src = APL_sb_from_src(sbs, src)

    rn_reg_t2 = BLEIR.RN_REG(f"rn_reg_t2")
    rn_reg_t5 = BLEIR.RN_REG(f"rn_reg_t5")
    rn_reg_t7 = BLEIR.RN_REG(f"rn_reg_t7")
    rn_registers = {
        2: rn_reg_t2.identifier,
        5: rn_reg_t5.identifier,
        7: rn_reg_t7.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t2.identifier] = rn_reg_t2
    bleir_map[rn_reg_t5.identifier] = rn_reg_t5
    bleir_map[rn_reg_t7.identifier] = rn_reg_t7

    expected_value = BLEIR.assign(BLEIR.SB[rn_reg_t2, rn_reg_t5, rn_reg_t7], BLEIR.RL)
    actual_value = sb_from_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)


def test_apl_sb_from_src_str():
    sbs = [1, 2, 3]
    src = "RL"
    sb_from_src = APL_sb_from_src(sbs, src)
    assert str(sb_from_src) == "SB[1,2,3] = RL"


def test_apl_set_rl_render_bleir():
    value = 1
    set_rl = APL_set_rl(value)

    rn_registers = {}
    bleir_map = make_bleir_map()

    expected_value = BLEIR.assign(BLEIR.RL, value)
    actual_value = set_rl.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)


def test_apl_set_rl_str():
    value = 0
    set_rl = APL_set_rl(value)
    assert str(set_rl) == "RL = 0"


def test_apl_rl_from_sb_render_bleir():

    # One RN_REG (=)
    # --------------

    sbs = [0]
    assign_op = ""
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_registers = {
        0: rn_reg_t0.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0

    expected_value = BLEIR.assign(BLEIR.RL, BLEIR.SB[rn_reg_t0])
    actual_value = rl_from_sb.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Two RN_REGs (=)
    # ---------------

    sbs = [0, 1]
    assign_op = ""
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1

    expected_value = BLEIR.assign(BLEIR.RL, BLEIR.SB[rn_reg_t0, rn_reg_t1])
    actual_value = rl_from_sb.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Three RN_REGs (=)
    # -----------------

    sbs = [0, 1, 2]
    assign_op = ""
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_reg_t2 = BLEIR.RN_REG("rn_reg_t2")
    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
        2: rn_reg_t2.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1
    bleir_map[rn_reg_t2.identifier] = rn_reg_t2

    expected_value = BLEIR.assign(BLEIR.RL, BLEIR.SB[rn_reg_t0, rn_reg_t1, rn_reg_t2])
    actual_value = rl_from_sb.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # One RN_REG (&=)
    # ---------------

    sbs = [0]
    assign_op = "&"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_registers = {
        0: rn_reg_t0.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0

    expected_value = BLEIR.and_eq(BLEIR.RL, BLEIR.SB[rn_reg_t0])
    actual_value = rl_from_sb.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Two RN_REGs (&=)
    # ----------------

    sbs = [0, 1]
    assign_op = "&"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1

    expected_value = BLEIR.and_eq(BLEIR.RL, BLEIR.SB[rn_reg_t0, rn_reg_t1])
    actual_value = rl_from_sb.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Three RN_REGs (&=)
    # ------------------

    sbs = [0, 1, 2]
    assign_op = "&"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_reg_t2 = BLEIR.RN_REG("rn_reg_t2")
    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
        2: rn_reg_t2.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1
    bleir_map[rn_reg_t2.identifier] = rn_reg_t2

    expected_value = BLEIR.and_eq(BLEIR.RL, BLEIR.SB[rn_reg_t0, rn_reg_t1, rn_reg_t2])
    actual_value = rl_from_sb.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Exception cases (&=)
    # --------------------

    # One RN_REG (|=)
    # ---------------

    sbs = [0]
    assign_op = "|"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_registers = {
        0: rn_reg_t0.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0

    expected_value = BLEIR.or_eq(BLEIR.RL, BLEIR.SB[rn_reg_t0])
    actual_value = rl_from_sb.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Two RN_REGs (|=)
    # ----------------

    sbs = [0, 1]
    assign_op = "|"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1

    expected_value = BLEIR.or_eq(BLEIR.RL, BLEIR.SB[rn_reg_t0, rn_reg_t1])
    actual_value = rl_from_sb.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Three RN_REGs (|=)
    # ------------------

    sbs = [0, 1, 2]
    assign_op = "|"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_reg_t2 = BLEIR.RN_REG("rn_reg_t2")
    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
        2: rn_reg_t2.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1
    bleir_map[rn_reg_t2.identifier] = rn_reg_t2

    expected_value = BLEIR.or_eq(BLEIR.RL, BLEIR.SB[rn_reg_t0, rn_reg_t1, rn_reg_t2])
    actual_value = rl_from_sb.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Exception cases (|=)
    # --------------------

    # One RN_REG (^=)
    # ---------------

    sbs = [0]
    assign_op = "^"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_registers = {
        0: rn_reg_t0.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0

    expected_value = BLEIR.xor_eq(BLEIR.RL, BLEIR.SB[rn_reg_t0])
    actual_value = rl_from_sb.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Two RN_REGs (^=)
    # ----------------

    sbs = [0, 1]
    assign_op = "^"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1

    expected_value = BLEIR.xor_eq(BLEIR.RL, BLEIR.SB[rn_reg_t0, rn_reg_t1])
    actual_value = rl_from_sb.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Three RN_REGs (^=)
    # ------------------

    sbs = [0, 1, 2]
    assign_op = "^"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_reg_t2 = BLEIR.RN_REG("rn_reg_t2")
    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
        2: rn_reg_t2.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1
    bleir_map[rn_reg_t2.identifier] = rn_reg_t2

    expected_value = BLEIR.xor_eq(BLEIR.RL, BLEIR.SB[rn_reg_t0, rn_reg_t1, rn_reg_t2])
    actual_value = rl_from_sb.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Exception cases (^=)
    # --------------------

    # Exception cases (?=)

    # "?=" is not a valid READ op for RL (only SB)
    with pytest.raises(BLEIR.SemanticError):
        sbs = [0]
        assign_op = "?"
        rl_from_sb = APL_rl_from_sb(sbs, assign_op)

        rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
        rn_registers = {
            0: rn_reg_t0.identifier,
        }

        bleir_map = make_bleir_map()
        bleir_map[rn_reg_t0.identifier] = rn_reg_t0

        actual_value = rl_from_sb.render_bleir(rn_registers, bleir_map)
        validate_types(actual_value)


def test_apl_rl_from_sb_str():
    sbs = [0]
    assign_op = ""
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    assert str(rl_from_sb) == "RL = SB[0]"

    sbs = [0, 1]
    assign_op = ""
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    assert str(rl_from_sb) == "RL = SB[0,1]"

    sbs = [0, 1, 2]
    assign_op = ""
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    assert str(rl_from_sb) == "RL = SB[0,1,2]"

    sbs = [0]
    assign_op = "&"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    assert str(rl_from_sb) == "RL &= SB[0]"

    sbs = [0, 1]
    assign_op = "&"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    assert str(rl_from_sb) == "RL &= SB[0,1]"

    sbs = [0, 1, 2]
    assign_op = "&"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    assert str(rl_from_sb) == "RL &= SB[0,1,2]"

    sbs = [0]
    assign_op = "|"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    assert str(rl_from_sb) == "RL |= SB[0]"

    sbs = [0, 1]
    assign_op = "|"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    assert str(rl_from_sb) == "RL |= SB[0,1]"

    sbs = [0, 1, 2]
    assign_op = "|"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    assert str(rl_from_sb) == "RL |= SB[0,1,2]"

    sbs = [0]
    assign_op = "^"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    assert str(rl_from_sb) == "RL ^= SB[0]"

    sbs = [0, 1]
    assign_op = "^"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    assert str(rl_from_sb) == "RL ^= SB[0,1]"

    sbs = [0, 1, 2]
    assign_op = "^"
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    assert str(rl_from_sb) == "RL ^= SB[0,1,2]"


def test_apl_rl_from_src_render_bleir():

    # (=)
    # ---

    src = "RSP16"
    assign_op = ""
    rl_from_src = APL_rl_from_src(src, assign_op)

    rn_registers = {}
    bleir_map = make_bleir_map()

    expected_value = BLEIR.assign(BLEIR.RL, BLEIR.RSP16)
    actual_value = rl_from_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # (&=)
    # ----

    src = "RSP16"
    assign_op = "&"
    rl_from_src = APL_rl_from_src(src, assign_op)

    rn_registers = {}
    bleir_map = make_bleir_map()

    expected_value = BLEIR.and_eq(BLEIR.RL, BLEIR.RSP16)
    actual_value = rl_from_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # (|=)
    # ----

    src = "RSP16"
    assign_op = "|"
    rl_from_src = APL_rl_from_src(src, assign_op)

    rn_registers = {}
    bleir_map = make_bleir_map()

    expected_value = BLEIR.or_eq(BLEIR.RL, BLEIR.RSP16)
    actual_value = rl_from_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # (^=)
    # ----

    src = "RSP16"
    assign_op = "^"
    rl_from_src = APL_rl_from_src(src, assign_op)

    rn_registers = {}
    bleir_map = make_bleir_map()

    expected_value = BLEIR.xor_eq(BLEIR.RL, BLEIR.RSP16)
    actual_value = rl_from_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Exception cases (?=)

    # "?=" is not a valid READ op for RL (only SB)
    with pytest.raises(BLEIR.SemanticError):
        src = "RSP16"
        assign_op = "?"
        rl_from_src = APL_rl_from_src(src, assign_op)

        rn_registers = {}
        bleir_map = make_bleir_map()

        actual_value = rl_from_src.render_bleir(rn_registers, bleir_map)
        validate_types(actual_value)


def test_apl_rl_from_src_str():
    src = "RSP16"
    assign_op = ""
    rl_from_src = APL_rl_from_src(src, assign_op)
    assert str(rl_from_src) == "RL = RSP16"

    src = "RSP16"
    assign_op = "&"
    rl_from_src = APL_rl_from_src(src, assign_op)
    assert str(rl_from_src) == "RL &= RSP16"

    src = "RSP16"
    assign_op = "|"
    rl_from_src = APL_rl_from_src(src, assign_op)
    assert str(rl_from_src) == "RL |= RSP16"

    src = "RSP16"
    assign_op = "^"
    rl_from_src = APL_rl_from_src(src, assign_op)
    assert str(rl_from_src) == "RL ^= RSP16"


def test_apl_rl_from_sb_src_render_bleir():

    # One RN_REG (=; &)
    # -----------------

    sbs = [0]
    src = "RSP16"
    assign_op = ""
    binary_op = "&"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")

    rn_registers = {
        0: rn_reg_t0.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0

    expected_value = BLEIR.assign(BLEIR.RL,
                                  BLEIR.conjoin(BLEIR.SB[rn_reg_t0],
                                                BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Two RN_REGs (=; &)
    # ------------------

    sbs = [0, 1]
    src = "RSP16"
    assign_op = ""
    binary_op = "&"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")

    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1

    expected_value = BLEIR.assign(BLEIR.RL,
                                  BLEIR.conjoin(BLEIR.SB[rn_reg_t0, rn_reg_t1],
                                                BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Three RN_REGs (=; &)
    # --------------------

    sbs = [0, 1, 2]
    src = "RSP16"
    assign_op = ""
    binary_op = "&"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_reg_t2 = BLEIR.RN_REG("rn_reg_t2")

    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
        2: rn_reg_t2.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1
    bleir_map[rn_reg_t2.identifier] = rn_reg_t2

    expected_value = BLEIR.assign(BLEIR.RL,
                                  BLEIR.conjoin(BLEIR.SB[rn_reg_t0, rn_reg_t1, rn_reg_t2],
                                                BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # One RN_REG (&=; &)
    # -----------------

    sbs = [0]
    src = "RSP16"
    assign_op = "&"
    binary_op = "&"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")

    rn_registers = {
        0: rn_reg_t0.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0

    expected_value = BLEIR.and_eq(BLEIR.RL,
                                  BLEIR.conjoin(BLEIR.SB[rn_reg_t0],
                                                BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Two RN_REGs (&=; &)
    # ------------------

    sbs = [0, 1]
    src = "RSP16"
    assign_op = "&"
    binary_op = "&"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")

    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1

    expected_value = BLEIR.and_eq(BLEIR.RL,
                                  BLEIR.conjoin(BLEIR.SB[rn_reg_t0, rn_reg_t1],
                                                BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Three RN_REGs (&=; &)
    # --------------------

    sbs = [0, 1, 2]
    src = "RSP16"
    assign_op = "&"
    binary_op = "&"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_reg_t2 = BLEIR.RN_REG("rn_reg_t2")

    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
        2: rn_reg_t2.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1
    bleir_map[rn_reg_t2.identifier] = rn_reg_t2

    expected_value = BLEIR.and_eq(BLEIR.RL,
                                  BLEIR.conjoin(BLEIR.SB[rn_reg_t0, rn_reg_t1, rn_reg_t2],
                                                BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # One RN_REG (|=; &)
    # -----------------

    sbs = [0]
    src = "RSP16"
    assign_op = "|"
    binary_op = "&"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")

    rn_registers = {
        0: rn_reg_t0.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0

    expected_value = BLEIR.or_eq(BLEIR.RL,
                                 BLEIR.conjoin(BLEIR.SB[rn_reg_t0],
                                               BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Two RN_REGs (|=; &)
    # ------------------

    sbs = [0, 1]
    src = "RSP16"
    assign_op = "|"
    binary_op = "&"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")

    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1

    expected_value = BLEIR.or_eq(BLEIR.RL,
                                 BLEIR.conjoin(BLEIR.SB[rn_reg_t0, rn_reg_t1],
                                               BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Three RN_REGs (|=; &)
    # --------------------

    sbs = [0, 1, 2]
    src = "RSP16"
    assign_op = "|"
    binary_op = "&"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_reg_t2 = BLEIR.RN_REG("rn_reg_t2")

    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
        2: rn_reg_t2.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1
    bleir_map[rn_reg_t2.identifier] = rn_reg_t2

    expected_value = BLEIR.or_eq(BLEIR.RL,
                                 BLEIR.conjoin(BLEIR.SB[rn_reg_t0, rn_reg_t1, rn_reg_t2],
                                               BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # One RN_REG (^=; &)
    # -----------------

    sbs = [0]
    src = "RSP16"
    assign_op = "^"
    binary_op = "&"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")

    rn_registers = {
        0: rn_reg_t0.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0

    expected_value = BLEIR.xor_eq(BLEIR.RL,
                                  BLEIR.conjoin(BLEIR.SB[rn_reg_t0],
                                                BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Two RN_REGs (^=; &)
    # ------------------

    sbs = [0, 1]
    src = "RSP16"
    assign_op = "^"
    binary_op = "&"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")

    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1

    expected_value = BLEIR.xor_eq(BLEIR.RL,
                                  BLEIR.conjoin(BLEIR.SB[rn_reg_t0, rn_reg_t1],
                                                BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Three RN_REGs (^=; &)
    # --------------------

    sbs = [0, 1, 2]
    src = "RSP16"
    assign_op = "^"
    binary_op = "&"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_reg_t2 = BLEIR.RN_REG("rn_reg_t2")

    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
        2: rn_reg_t2.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1
    bleir_map[rn_reg_t2.identifier] = rn_reg_t2

    expected_value = BLEIR.xor_eq(BLEIR.RL,
                                  BLEIR.conjoin(BLEIR.SB[rn_reg_t0, rn_reg_t1, rn_reg_t2],
                                                BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # One RN_REG (=; |)
    # -----------------

    sbs = [0]
    src = "RSP16"
    assign_op = ""
    binary_op = "|"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")

    rn_registers = {
        0: rn_reg_t0.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0

    expected_value = BLEIR.assign(BLEIR.RL,
                                  BLEIR.disjoin(BLEIR.SB[rn_reg_t0],
                                                BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Two RN_REGs (=; |)
    # ------------------

    sbs = [0, 1]
    src = "RSP16"
    assign_op = ""
    binary_op = "|"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")

    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1

    expected_value = BLEIR.assign(BLEIR.RL,
                                  BLEIR.disjoin(BLEIR.SB[rn_reg_t0, rn_reg_t1],
                                                BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Three RN_REGs (=; |)
    # --------------------

    sbs = [0, 1, 2]
    src = "RSP16"
    assign_op = ""
    binary_op = "|"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_reg_t2 = BLEIR.RN_REG("rn_reg_t2")

    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
        2: rn_reg_t2.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1
    bleir_map[rn_reg_t2.identifier] = rn_reg_t2

    expected_value = BLEIR.assign(BLEIR.RL,
                                  BLEIR.disjoin(BLEIR.SB[rn_reg_t0, rn_reg_t1, rn_reg_t2],
                                                BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # One RN_REG (&=; |)
    # -----------------

    with pytest.raises(BLEIR.SemanticError):
        sbs = [0]
        src = "RSP16"
        assign_op = "&"
        binary_op = "|"
        rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

        rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")

        rn_registers = {
            0: rn_reg_t0.identifier,
        }

        bleir_map = make_bleir_map()
        bleir_map[rn_reg_t0.identifier] = rn_reg_t0

        rl_from_sb_src.render_bleir(rn_registers, bleir_map)

    # One RN_REG (|=; |)
    # -----------------

    with pytest.raises(BLEIR.SemanticError):
        sbs = [0]
        src = "RSP16"
        assign_op = "|"
        binary_op = "|"
        rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

        rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")

        rn_registers = {
            0: rn_reg_t0.identifier,
        }

        bleir_map = make_bleir_map()
        bleir_map[rn_reg_t0.identifier] = rn_reg_t0

        rl_from_sb_src.render_bleir(rn_registers, bleir_map)

    # One RN_REG (^=; |)
    # -----------------

    with pytest.raises(BLEIR.SemanticError):
        sbs = [0]
        src = "RSP16"
        assign_op = "^"
        binary_op = "|"
        rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

        rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")

        rn_registers = {
            0: rn_reg_t0.identifier,
        }

        bleir_map = make_bleir_map()
        bleir_map[rn_reg_t0.identifier] = rn_reg_t0

        rl_from_sb_src.render_bleir(rn_registers, bleir_map)

    # One RN_REG (=; ^)
    # -----------------

    sbs = [0]
    src = "RSP16"
    assign_op = ""
    binary_op = "^"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")

    rn_registers = {
        0: rn_reg_t0.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0

    expected_value = BLEIR.assign(BLEIR.RL,
                                  BLEIR.xor(BLEIR.SB[rn_reg_t0],
                                            BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Two RN_REGs (=; ^)
    # ------------------

    sbs = [0, 1]
    src = "RSP16"
    assign_op = ""
    binary_op = "^"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")

    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1

    expected_value = BLEIR.assign(BLEIR.RL,
                                  BLEIR.xor(BLEIR.SB[rn_reg_t0, rn_reg_t1],
                                            BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # Three RN_REGs (=; ^)
    # --------------------

    sbs = [0, 1, 2]
    src = "RSP16"
    assign_op = ""
    binary_op = "^"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_reg_t2 = BLEIR.RN_REG("rn_reg_t2")

    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
        2: rn_reg_t2.identifier,
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1
    bleir_map[rn_reg_t2.identifier] = rn_reg_t2

    expected_value = BLEIR.assign(BLEIR.RL,
                                  BLEIR.xor(BLEIR.SB[rn_reg_t0, rn_reg_t1, rn_reg_t2],
                                            BLEIR.RSP16))
    actual_value = rl_from_sb_src.render_bleir(rn_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)

    # One RN_REG (&=; ^)
    # -----------------

    with pytest.raises(BLEIR.SemanticError):
        sbs = [0]
        src = "RSP16"
        assign_op = "&"
        binary_op = "^"
        rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

        rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")

        rn_registers = {
            0: rn_reg_t0.identifier,
        }

        bleir_map = make_bleir_map()
        bleir_map[rn_reg_t0.identifier] = rn_reg_t0

        rl_from_sb_src.render_bleir(rn_registers, bleir_map)

    # One RN_REG (|=; ^)
    # -----------------

    with pytest.raises(BLEIR.SemanticError):
        sbs = [0]
        src = "RSP16"
        assign_op = "|"
        binary_op = "^"
        rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

        rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")

        rn_registers = {
            0: rn_reg_t0.identifier,
        }

        bleir_map = make_bleir_map()
        bleir_map[rn_reg_t0.identifier] = rn_reg_t0

        rl_from_sb_src.render_bleir(rn_registers, bleir_map)

    # One RN_REG (^=; ^)
    # -----------------

    with pytest.raises(BLEIR.SemanticError):
        sbs = [0]
        src = "RSP16"
        assign_op = "^"
        binary_op = "^"
        rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)

        rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")

        rn_registers = {
            0: rn_reg_t0.identifier,
        }

        bleir_map = make_bleir_map()
        bleir_map[rn_reg_t0.identifier] = rn_reg_t0

        rl_from_sb_src.render_bleir(rn_registers, bleir_map)


def test_apl_rl_from_sb_src_str():
    sbs = [0]
    src = "RSP16"
    assign_op = "&"
    binary_op = "&"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)
    assert str(rl_from_sb_src) == "RL &= SB[0] & RSP16"

    sbs = [0, 1]
    src = "NRL"
    assign_op = ""
    binary_op = "|"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)
    assert str(rl_from_sb_src) == "RL = SB[0,1] | NRL"

    sbs = [0, 1, 2]
    src = "SRL"
    assign_op = "|"
    binary_op = "&"
    rl_from_sb_src = APL_rl_from_sb_src(sbs, src, assign_op, binary_op)
    assert str(rl_from_sb_src) == "RL |= SB[0,1,2] & SRL"


def test_collect_all_normalized_masks():
    masked_stmts = []

    sbs = [0, 1, 2]
    assign_op = ""

    mask = Mask(to_bit_indices(0xFFFF))
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    masked_stmt = APL_masked_stmt(mask, rl_from_sb)
    masked_stmts.append(masked_stmt)

    mask = Mask(to_bit_indices(0xFFF0))
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    masked_stmt = APL_masked_stmt(mask, rl_from_sb)
    masked_stmts.append(masked_stmt)

    mask = Mask(to_bit_indices(0xFF00))
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    masked_stmt = APL_masked_stmt(mask, rl_from_sb)
    masked_stmts.append(masked_stmt)

    mask = Mask(to_bit_indices(0xF000))
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    masked_stmt = APL_masked_stmt(mask, rl_from_sb)
    masked_stmts.append(masked_stmt)

    mask = Mask(to_bit_indices(0x0101))
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    masked_stmt = APL_masked_stmt(mask, rl_from_sb)
    masked_stmts.append(masked_stmt)

    mask = Mask(to_bit_indices(0x1010))
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    masked_stmt = APL_masked_stmt(mask, rl_from_sb)
    masked_stmts.append(masked_stmt)

    mask = Mask(to_bit_indices(0x0001))
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    masked_stmt = APL_masked_stmt(mask, rl_from_sb)
    masked_stmts.append(masked_stmt)

    mask = Mask(to_bit_indices(0x0010))
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    masked_stmt = APL_masked_stmt(mask, rl_from_sb)
    masked_stmts.append(masked_stmt)

    mask = Mask(to_bit_indices(0x0100))
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    masked_stmt = APL_masked_stmt(mask, rl_from_sb)
    masked_stmts.append(masked_stmt)

    mask = Mask(to_bit_indices(0x1000))
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)
    masked_stmt = APL_masked_stmt(mask, rl_from_sb)
    masked_stmts.append(masked_stmt)

    used_masks = collect_all_normalized_masks_from_masked_stmt(masked_stmts)
    assert used_masks == {"0xffff", "0x0101", "0x0001"}


def test_apl_masked_stmt_render_bleir():
    sbs = [0, 1, 2]
    assign_op = ""
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)

    mask = Mask(to_bit_indices(0xABCD))
    masked_stmt = APL_masked_stmt(mask, rl_from_sb)

    rn_reg_t0 = BLEIR.RN_REG("rn_reg_t0")
    rn_reg_t1 = BLEIR.RN_REG("rn_reg_t1")
    rn_reg_t2 = BLEIR.RN_REG("rn_reg_t2")
    rn_registers = {
        0: rn_reg_t0.identifier,
        1: rn_reg_t1.identifier,
        2: rn_reg_t2.identifier,
    }

    used_masks = collect_all_normalized_masks_from_masked_stmt([masked_stmt])
    sm_registers = {
        used_mask: f"SM_{used_mask.upper()}"
        for used_mask in used_masks
    }

    bleir_map = make_bleir_map()
    bleir_map[rn_reg_t0.identifier] = rn_reg_t0
    bleir_map[rn_reg_t1.identifier] = rn_reg_t1
    bleir_map[rn_reg_t2.identifier] = rn_reg_t2

    for sm_reg in sm_registers.values():
        sm_reg_param = BLEIR.SM_REG(sm_reg)
        bleir_map[sm_reg] = sm_reg_param

    expected_value = BLEIR.STATEMENT(
        operation=BLEIR.MASKED(
            mask=BLEIR.MASK(
                expression=BLEIR.SHIFTED_SM_REG(
                    register=BLEIR.SM_REG(identifier="SM_0XABCD"),
                    num_bits=0)),
            assignment=BLEIR.ASSIGNMENT(
                operation=BLEIR.READ(
                    operator=BLEIR.ASSIGN_OP.EQ,
                    rvalue=BLEIR.UNARY_EXPR(
                        expression=BLEIR.UNARY_SB(
                            expression=BLEIR.SB_EXPR(
                                parameters=(rn_reg_t0, rn_reg_t1, rn_reg_t2))))))))

    actual_value = masked_stmt.render_bleir(rn_registers, sm_registers, bleir_map)
    assert expected_value == actual_value
    validate_types(actual_value)


def test_apl_masked_stmt_str():
    sbs = [0, 1, 2]
    assign_op = ""
    rl_from_sb = APL_rl_from_sb(sbs, assign_op)

    mask = Mask(to_bit_indices(0xABCD))
    masked_stmt = APL_masked_stmt(mask, rl_from_sb)
    assert str(masked_stmt) == "0xabcd : RL = SB[0,1,2];"
