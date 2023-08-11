r"""By Dylon Edwards

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

from copy import deepcopy

import numpy as np

import pytest

import hypothesis
from hypothesis import given

import open_belex.bleir.types as BLEIR
import open_belex.common.mask
from open_belex.common.constants import NSB, NSECTIONS, NUM_PLATS_PER_APUC
from open_belex.common.seu_layer import SEULayer
from open_belex.diri.half_bank import DIRI, GGL_SHAPE
from open_belex.kernel_libs.memory import (copy_l1_to_l2_byte,
                                           copy_l2_to_l1_byte, l2_end,
                                           load_16_t0, store_16_t0)
from open_belex.literal import (ASSIGN_OP, BINOP, ERL, EWE, EWE_REG_0,
                                EWE_REG_1, EWE_REG_2, EWE_REG_3, FSEL_NOOP,
                                GGL, GL, INV_ERL, INV_GGL, INV_GL, INV_NRL,
                                INV_RL, INV_RSP16, INV_SRL, INV_WRL, L1,
                                L1_ADDR_REG_0, L1_ADDR_REG_1, L1_ADDR_REG_2,
                                L1_ADDR_REG_3, L2, L2_ADDR_REG_0, LGL, NOOP,
                                NRL, RE, RE_REG_0, RE_REG_1, RE_REG_2,
                                RE_REG_3, RL, RN_REG_0, RN_REG_1, RN_REG_2,
                                RN_REG_3, RN_REG_4, RN_REG_5, RN_REG_6,
                                RN_REG_7, RN_REG_8, RN_REG_9, RN_REG_10,
                                RN_REG_11, RN_REG_12, RN_REG_13, RN_REG_14,
                                RN_REG_15, RN_REG_T0, RSP2K, RSP16, RSP32K,
                                RSP256, RSP_END, RSP_START_RET, RWINH_RST,
                                RWINH_SET, SM_0X000F, SM_0X0001, SM_0X1111,
                                SM_0XFFFF, SM_REG_0, SM_REG_1, SM_REG_2,
                                SM_REG_3, SM_REG_4, SM_REG_5, SM_REG_6,
                                SM_REG_7, SM_REG_8, SM_REG_9, SM_REG_10,
                                SM_REG_11, SM_REG_12, SM_REG_13, SM_REG_14,
                                SM_REG_15, SRL, VR, WRL, AssignOperation,
                                Belex, BelexAccess, BinaryOperation, Mask,
                                NoopInstruction, OffsetLX, RspEndInstruction,
                                Section, apl_commands, apl_set_ewe_reg,
                                apl_set_l1_reg, apl_set_l1_reg_ext,
                                apl_set_l2_reg, apl_set_re_reg, apl_set_rn_reg,
                                apl_set_sm_reg, append_commands,
                                appendable_commands, belex_apl, u16)
from open_belex.utils.example_utils import (convert_to_bool, convert_to_u16,
                                            u16_to_bool)

from open_belex_libs.common import cpy_16, cpy_imm_16, reset_16
from open_belex_libs.memory import (belex_gal_vm_reg_to_set_ext,
                                    load_16_parity_mask, store_16_parity_mask)

from open_belex_tests.utils import (Mask_strategy, c_caller,
                                    parameterized_belex_test, seu_context,
                                    vr_strategy)


def test_accessibility():
    out = VR("out")
    mask = Mask("SM_0XBEEF")
    for var in [out, RL]:
        assert var() == BelexAccess(var=var, mask=None)
        assert var[mask] == BelexAccess(var=var, mask=mask)

    assert not out().is_negated
    assert ~out().is_negated


def test_validity():
    for sb in range(NSB):
        VR.validate(sb)

    with pytest.raises(ValueError):
        VR.validate(-1)

    with pytest.raises(ValueError):
        VR.validate(NSB + 1)

    for section_mask in range(0xFFFF):
        Mask.validate(section_mask)

    with pytest.raises(ValueError):
        Mask.validate(-1)

    with pytest.raises(ValueError):
        Mask.validate(0xFFFF + 1)

    for section in range(NSECTIONS):
        Section.validate(section)

    with pytest.raises(ValueError):
        Section.validate(-1)

    with pytest.raises(ValueError):
        Section.validate(NSECTIONS + 1)

    with pytest.raises(ValueError):
        Section.validate(0x1001)

    with pytest.raises(ValueError):
        Section.validate(0xFFFF)


def test_mask_ops():
    mask = Mask("SM_0XFFFF")
    assert not mask.is_negated
    assert mask.shift_width == 0

    mask = (mask << 0)
    assert not mask.is_negated
    assert mask.shift_width == 0

    mask = (mask << 1)
    assert not mask.is_negated
    assert mask.shift_width == 1

    mask = (mask << 3)
    assert not mask.is_negated
    assert mask.shift_width == 4

    mask = ~mask
    assert mask.is_negated
    assert mask.shift_width == 4

    mask = ~mask
    assert not mask.is_negated
    assert mask.shift_width == 4

    mask = (~mask << 2)
    assert mask.is_negated
    assert mask.shift_width == 6

    with pytest.raises(ValueError):
        mask << -1

    with pytest.raises(ValueError):
        mask << NSECTIONS


def test_mask_literals():
    belex = Belex.push_context()

    try:
        sm_0x1111 = belex.Mask(0x1111)
        assert sm_0x1111.constant_value == 0x1111

        sm_0x0001 = belex.Section(0)

        lvr = VR("lvr")

        assert lvr[[3, 7, 11, 15]] == lvr[sm_0x1111<<3]
        assert lvr["37BF"] == lvr[sm_0x1111<<3]
        assert lvr["0x8888"] == lvr[sm_0x1111<<3]

        assert lvr[0] == lvr[sm_0x0001]
        assert lvr[[0]] == lvr[sm_0x0001]
        assert lvr["0"] == lvr[sm_0x0001]
        assert lvr["0x0001"] == lvr[sm_0x0001]
    finally:
        Belex.pop_context()
        del belex


def test_conjuntibility():
    lvr = VR("lvr")

    assert lvr() & RL() == \
        BinaryOperation(BINOP.AND, lvr(), RL())

    assert ~lvr() & RL() == \
        BinaryOperation(BINOP.AND, ~lvr(), RL())

    assert lvr() & ~RL() == \
        BinaryOperation(BINOP.AND, lvr(), ~RL())

    assert ~lvr() & ~RL() == \
        BinaryOperation(BINOP.AND, ~lvr(), ~RL())

    assert lvr() | RL() == \
        BinaryOperation(BINOP.OR, lvr(), RL())

    assert lvr() ^ RL() == \
        BinaryOperation(BINOP.XOR, lvr(), RL())


def test_assignability():
    belex = Belex.push_context()

    try:
        msk = Mask("SM_0xF00D")
        assert msk.as_bleir() == BLEIR.mask("SM_0xF00D")

        lvr = VR("lvr")
        rvr = VR("rvr")
        rvr2 = VR("rvr2")
        assert lvr.as_bleir() == BLEIR.RN_REG("lvr")

        lvr[msk] <= RL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, lvr[msk], RL())
        assert belex.instructions[-1].as_patterns() == ["SB = <SRC>", "SB = RL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.assign(
                                             BLEIR.SB[lvr.as_bleir()],
                                             RL.as_bleir()))),
        ]

        lvr[msk] <= ~RL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, lvr[msk], ~RL())
        assert belex.instructions[-1].as_patterns() == ["SB = ~<SRC>", "SB = ~RL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.assign(
                                             BLEIR.SB[lvr.as_bleir()],
                                             ~RL.as_bleir()))),
        ]

        lvr[msk] |= GGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.COND_EQ, lvr[msk], GGL())
        assert belex.instructions[-1].as_patterns() == ["SB ?= <SRC>", "SB ?= GGL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.cond_eq(
                                             BLEIR.SB[lvr.as_bleir()],
                                             GGL.as_bleir()))),
        ]

        RL[msk] <= 1
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RL[msk], 1)
        assert belex.instructions[-1].as_patterns() == ["RL = <BIT>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.assign(
                                             RL.as_bleir(),
                                             1))),
        ]

        RL[~msk] <= 0
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RL[~msk], 0)
        assert belex.instructions[-1].as_patterns() == ["RL = <BIT>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(~msk.as_bleir(),
                                         BLEIR.assign(
                                             RL.as_bleir(),
                                             0))),
        ]

        RL[msk] <= lvr()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RL[msk], lvr())
        assert belex.instructions[-1].as_patterns() == ["RL = <SB>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.assign(
                                             RL.as_bleir(),
                                             BLEIR.SB[lvr.as_bleir()]))),
        ]

        RL[msk] <= RSP16()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RL[msk], RSP16())
        assert belex.instructions[-1].as_patterns() == ["RL = <SRC>", "RL = RSP16"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.assign(
                                             RL.as_bleir(),
                                             RSP16.as_bleir()))),
        ]

        RL[msk] <= lvr() & GL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RL[msk],
                            BinaryOperation(BINOP.AND, lvr(), GL()))
        assert belex.instructions[-1].as_patterns() == ["RL = (<SB> & <SRC>)", "RL = (<SB> & GL)"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.assign(
                                             RL.as_bleir(),
                                             BLEIR.conjoin(
                                                 BLEIR.SB[lvr.as_bleir()],
                                                 GL.as_bleir())))),
        ]

        RL[msk] |= lvr()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.OR_EQ, RL[msk], lvr())
        assert belex.instructions[-1].as_patterns() == ["RL |= <SB>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.or_eq(
                                             RL.as_bleir(),
                                             BLEIR.SB[lvr.as_bleir()]))),
        ]

        RL[msk] |= RSP16()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.OR_EQ, RL[msk], RSP16())
        assert belex.instructions[-1].as_patterns() == ["RL |= <SRC>", "RL |= RSP16"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.or_eq(
                                             RL.as_bleir(),
                                             RSP16.as_bleir()))),
        ]

        RL[msk] |= lvr() & GGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.OR_EQ, RL[msk],
                            BinaryOperation(BINOP.AND, lvr(), GGL()))
        assert belex.instructions[-1].as_patterns() == ["RL |= (<SB> & <SRC>)", "RL |= (<SB> & GGL)"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.or_eq(
                                             RL.as_bleir(),
                                             BLEIR.conjoin(
                                                 BLEIR.SB[lvr.as_bleir()],
                                                 GGL.as_bleir())))),
        ]

        RL[msk] &= lvr()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.AND_EQ, RL[msk], lvr())
        assert belex.instructions[-1].as_patterns() == ["RL &= <SB>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.and_eq(
                                             RL.as_bleir(),
                                             BLEIR.SB[lvr.as_bleir()]))),
        ]

        RL[msk] &= RSP16()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.AND_EQ, RL[msk], RSP16())
        assert belex.instructions[-1].as_patterns() == ["RL &= <SRC>", "RL &= RSP16"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.and_eq(
                                             RL.as_bleir(),
                                             RSP16.as_bleir()))),
        ]

        RL[msk] &= lvr() & NRL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.AND_EQ, RL[msk],
                            BinaryOperation(BINOP.AND, lvr(), NRL()))
        assert belex.instructions[-1].as_patterns() == ["RL &= (<SB> & <SRC>)", "RL &= (<SB> & NRL)"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.and_eq(
                                             RL.as_bleir(),
                                             BLEIR.conjoin(
                                                 BLEIR.SB[lvr.as_bleir()],
                                                 NRL.as_bleir())))),
        ]

        RL[msk] ^= lvr()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.XOR_EQ, RL[msk], lvr())
        assert belex.instructions[-1].as_patterns() == ["RL ^= <SB>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.xor_eq(
                                             RL.as_bleir(),
                                             BLEIR.SB[lvr.as_bleir()]))),
        ]

        RL[msk] ^= RSP16()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.XOR_EQ, RL[msk], RSP16())
        assert belex.instructions[-1].as_patterns() == ["RL ^= <SRC>", "RL ^= RSP16"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.xor_eq(
                                             RL.as_bleir(),
                                             RSP16.as_bleir()))),
        ]

        RL[msk] ^= lvr() & RSP16()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.XOR_EQ, RL[msk],
                            BinaryOperation(BINOP.AND, lvr(), RSP16()))
        assert belex.instructions[-1].as_patterns() == ["RL ^= (<SB> & <SRC>)", "RL ^= (<SB> & RSP16)"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.xor_eq(
                                             RL.as_bleir(),
                                             BLEIR.conjoin(
                                                 BLEIR.SB[lvr.as_bleir()],
                                                 RSP16.as_bleir())))),
        ]

        RL[msk] <= lvr() | GL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RL[msk],
                            BinaryOperation(BINOP.OR, lvr(), GL()))
        assert belex.instructions[-1].as_patterns() == ["RL = (<SB> | <SRC>)", "RL = (<SB> | GL)"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.assign(
                                             RL.as_bleir(),
                                             BLEIR.disjoin(
                                                 BLEIR.SB[lvr.as_bleir()],
                                                 GL.as_bleir())))),
        ]

        RL[msk] <= lvr() ^ RSP16()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RL[msk],
                            BinaryOperation(BINOP.XOR, lvr(), RSP16()))
        assert belex.instructions[-1].as_patterns() == ["RL = (<SB> ^ <SRC>)", "RL = (<SB> ^ RSP16)"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.assign(
                                             RL.as_bleir(),
                                             BLEIR.xor(
                                                 BLEIR.SB[lvr.as_bleir()],
                                                 RSP16.as_bleir())))),
        ]

        RL[msk] <= ~lvr() & NRL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RL[msk],
                            BinaryOperation(BINOP.AND, ~lvr(), NRL()))
        assert belex.instructions[-1].as_patterns() == ["RL = (~<SB> & <SRC>)", "RL = (~<SB> & NRL)"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.assign(
                                             RL.as_bleir(),
                                             BLEIR.conjoin(
                                                 ~BLEIR.SB[lvr.as_bleir()],
                                                 NRL.as_bleir())))),
        ]

        RL[msk] <= lvr() & ~SRL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RL[msk],
                            BinaryOperation(BINOP.AND, lvr(), ~SRL()))
        assert belex.instructions[-1].as_patterns() == ["RL = (<SB> & ~<SRC>)", "RL = (<SB> & ~SRL)"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.assign(
                                             RL.as_bleir(),
                                             BLEIR.conjoin(
                                                 BLEIR.SB[lvr.as_bleir()],
                                                 BLEIR.invert(SRL.as_bleir()))))),
        ]

        RL[msk] &= ~lvr()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.AND_EQ, RL[msk], ~lvr())
        assert belex.instructions[-1].as_patterns() == ["RL &= ~<SB>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.and_eq(RL.as_bleir(),
                                                      ~BLEIR.SB[lvr.as_bleir()]))),
        ]

        RL[msk] &= ~GGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.AND_EQ, RL[msk], ~GGL())
        assert belex.instructions[-1].as_patterns() == ["RL &= ~<SRC>", "RL &= ~GGL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.and_eq(RL.as_bleir(),
                                                      ~GGL.as_bleir()))),
        ]

        GL[msk] <= RL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, GL[msk], RL())
        assert belex.instructions[-1].as_patterns() == ["GL = <SRC>", "GL = RL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.assign(GL.as_bleir(),
                                                      RL.as_bleir()))),
        ]

        GGL[msk] <= RL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, GGL[msk], RL())
        assert belex.instructions[-1].as_patterns() == ["GGL = <SRC>", "GGL = RL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.assign(GGL.as_bleir(),
                                                      RL.as_bleir()))),
        ]

        RSP16[msk] <= RL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RSP16[msk], RL())
        assert belex.instructions[-1].as_patterns() == ["RSP16 = <SRC>", "RSP16 = RL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(msk.as_bleir(),
                                         BLEIR.assign(RSP16.as_bleir(),
                                                      RL.as_bleir()))),
        ]

        RSP16() <= RSP256()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RSP16(), RSP256())
        assert belex.instructions[-1].as_patterns() == ["RSP16 = RSP256"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(RSP16.as_bleir(),
                                         RSP256.as_bleir())),
        ]

        RSP256() <= RSP16()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RSP256(), RSP16())
        assert belex.instructions[-1].as_patterns() == ["RSP256 = <SRC>", "RSP256 = RSP16"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(RSP256.as_bleir(),
                                         RSP16.as_bleir())),
        ]

        RSP256() <= RSP2K()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RSP256(), RSP2K())
        assert belex.instructions[-1].as_patterns() == ["RSP256 = RSP2K"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(RSP256.as_bleir(),
                                         RSP2K.as_bleir())),
        ]

        RSP2K() <= RSP256()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RSP2K(), RSP256())
        assert belex.instructions[-1].as_patterns() == ["RSP2K = RSP256"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(RSP2K.as_bleir(),
                                         RSP256.as_bleir())),
        ]

        RSP2K() <= RSP32K()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RSP2K(), RSP32K())
        assert belex.instructions[-1].as_patterns() == ["RSP2K = RSP32K"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(RSP2K.as_bleir(),
                                         RSP32K.as_bleir())),
        ]

        RSP32K() <= RSP2K()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RSP32K(), RSP2K())
        assert belex.instructions[-1].as_patterns() == ["RSP32K = RSP2K"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(RSP32K.as_bleir(),
                                         RSP2K.as_bleir())),
        ]

        NOOP()
        assert belex.instructions[-1] == NoopInstruction()
        assert belex.instructions[-1].as_bleir() == \
            BLEIR.statement(BLEIR.SPECIAL.NOOP)

        RSP_END()
        assert belex.instructions[-1] == RspEndInstruction()
        assert belex.instructions[-1].as_bleir() == \
            BLEIR.statement(BLEIR.SPECIAL.RSP_END)

        with pytest.raises(AssertionError):
            RL[msk] <= lvr[msk<<1]
            belex.instructions[-1].as_bleir()

        with pytest.raises(AssertionError):
            RL[msk] <= lvr[:]
            belex.instructions[-1].as_bleir()

        with pytest.raises(AssertionError):
            RL[msk] <= lvr[::]
            belex.instructions[-1].as_bleir()

    finally:
        Belex.pop_context()


@belex_apl
def rsp32k_from_sb_wo_laning(Belex, dst: VR, src: VR) -> None:
    RL[::] <= src()
    RSP16[::] <= RL()
    RSP256() <= RSP16()
    RSP2K() <= RSP256()
    RSP32K() <= RSP2K()
    RSP_START_RET()
    RSP2K() <= RSP32K()
    RSP256() <= RSP2K()
    RSP16() <= RSP256()
    RL[::] <= RSP16()
    RSP_END()
    dst[::] <= RL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_rsp32k_from_sb_without_laning(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1

    expected_value = np.logical_or.reduce(diri.hb[src_vp], axis=0)
    expected_value = np.tile(expected_value, (NUM_PLATS_PER_APUC, 1))
    rsp32k_from_sb_wo_laning(dst_vp, src_vp)
    assert np.array_equal(expected_value, diri.hb[dst_vp])
    assert not diri.RSP32K.any()

    return dst_vp


@belex_apl
def rsp32k_from_sb_w_laning(Belex, dst: VR, src: VR) -> None:
    with apl_commands():
        RL[::] <= src()
        RSP16[::] <= RL()
    RSP256() <= RSP16()
    RSP2K() <= RSP256()
    RSP32K() <= RSP2K()
    RSP_START_RET()
    RSP2K() <= RSP32K()
    RSP256() <= RSP2K()
    RSP16() <= RSP256()
    RL[::] <= RSP16()
    RSP_END()
    dst[::] <= RL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_rsp32k_from_sb_with_laning(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1

    expected_value = np.logical_or.reduce(diri.hb[src_vp], axis=0)
    expected_value = np.tile(expected_value, (NUM_PLATS_PER_APUC, 1))
    rsp32k_from_sb_w_laning(dst_vp, src_vp)
    assert np.array_equal(expected_value, diri.hb[dst_vp])
    assert not diri.RSP32K.any()

    return dst_vp


@belex_apl
def get_initial_value(Belex, tgt: VR) -> None:
    sm_0xFFFF = Belex.Mask(0xFFFF)
    src = Belex.VR(0x1234)

    RL[sm_0xFFFF] <= src()
    tgt[sm_0xFFFF] <= RL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_get_initial_value(diri: DIRI) -> None:
    tgt_vp = 0
    fragment_caller_call = get_initial_value(tgt_vp)
    fragment = fragment_caller_call.fragment

    assert "\n".join(map(str, fragment.operations)) == "\n".join([
        "_INTERNAL_SM_0XFFFF: RL = SB[_INTERNAL_VR_000];",
        "_INTERNAL_SM_0XFFFF: SB[tgt] = RL;",
    ])

    tgt_vr = convert_to_u16(diri.hb[tgt_vp])
    assert all(tgt_vr == 0x1234)


@belex_apl
def mv_with_tmp(Belex, tgt: VR, src: VR) -> None:
    sm_0xFFFF = Belex.Mask(0xFFFF)
    tmp1 = Belex.VR()        # Uninitialized
    tmp2 = Belex.VR(0xBEEF)  # Initial Value

    RL[sm_0xFFFF] <= src()
    tmp1[sm_0xFFFF] <= RL()
    RL[sm_0xFFFF] <= 0
    RL[sm_0xFFFF] <= tmp1()
    tmp2[sm_0xFFFF] <= RL()
    RL[sm_0xFFFF] <= 0
    RL[sm_0xFFFF] <= tmp2()
    tgt[sm_0xFFFF] <= RL()


@parameterized_belex_test(repeatably_randomize_half_bank=True,
                          generate_code=False)
def test_mv_with_tmp(diri: DIRI) -> None:
    src_vp = 1
    tgt_vp = 0

    src_vr = convert_to_u16(diri.hb[src_vp])
    tgt_vr = convert_to_u16(diri.hb[tgt_vp])
    assert not np.array_equal(tgt_vr, src_vr)

    fragment_caller_call = mv_with_tmp(tgt_vp, src_vp)
    fragment = fragment_caller_call.fragment

    assert "\n".join(map(str, fragment.operations)) == "\n".join([
        "_INTERNAL_SM_0XFFFF: RL = SB[src];",
        "_INTERNAL_SM_0XFFFF: SB[_INTERNAL_VR_000] = RL;",
        "_INTERNAL_SM_0XFFFF: RL = 0;",
        "_INTERNAL_SM_0XFFFF: RL = SB[_INTERNAL_VR_000];",
        "_INTERNAL_SM_0XFFFF: SB[_INTERNAL_VR_001] = RL;",
        "_INTERNAL_SM_0XFFFF: RL = 0;",
        "_INTERNAL_SM_0XFFFF: RL = SB[_INTERNAL_VR_001];",
        "_INTERNAL_SM_0XFFFF: SB[tgt] = RL;",
    ])

    tgt_vr = convert_to_u16(diri.hb[tgt_vp])
    assert np.array_equal(tgt_vr, src_vr)


def test_ggl_assignments() -> None:
    belex = Belex.push_context()

    try:
        sm_0xffff = SM_0XFFFF.having(
            symbol="_INTERNAL_SM_0XFFFF",
            is_internal=True,
            is_lowered=False,
            register=None)

        sm_0x1111 = SM_0X1111.having(
            symbol="_INTERNAL_SM_0X1111",
            is_internal=True,
            is_lowered=False,
            register=None)

        GGL[::] <= RL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, GGL[sm_0xffff], RL())
        assert belex.instructions[-1].as_patterns() == ["GGL = <SRC>", "GGL = RL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(sm_0xffff, BLEIR.assign(GGL, RL))),
        ]

        l1 = L1(symbol="l1")
        l2 = L2(symbol="l2")

        GGL() <= l1()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, GGL(), l1())
        assert belex.instructions[-1].as_patterns() == ["GGL = <LX>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(GGL, l1)),
        ]

        GGL() <= l2()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, GGL(), l2())
        assert belex.instructions[-1].as_patterns() == ["GGL = <LX>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(GGL, l2)),
        ]

        GGL() <= l1() + 1
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, GGL(), l1() + 1)
        assert belex.instructions[-1].as_patterns() == ["GGL = <LX>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(GGL, BLEIR.offset(l1, 1))),
        ]

        GGL["0x1111"] <= RL() & l1()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, GGL[sm_0x1111],
                            BinaryOperation(BINOP.AND, RL(), l1()))
        assert belex.instructions[-1].as_patterns() == ["GGL = (<SRC> & <LX>)", "GGL = (RL & <LX>)"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.masked(sm_0x1111,
                                         BLEIR.assign(GGL, BLEIR.conjoin(RL, l1)))),
        ]
    finally:
        Belex.pop_context()


def test_lgl_assignments() -> None:
    belex = Belex.push_context()

    try:
        l1 = L1(symbol="l1")
        l2 = L2(symbol="l2")

        LGL() <= l1()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, LGL(), l1())
        assert belex.instructions[-1].as_patterns() == ["LGL = <LX>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(LGL, l1)),
        ]

        LGL() <= l2()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, LGL(), l2())
        assert belex.instructions[-1].as_patterns() == ["LGL = <LX>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(LGL, l2)),
        ]

        LGL() <= l1() + 1
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, LGL(), l1() + 1)
        assert belex.instructions[-1].as_patterns() == ["LGL = <LX>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(LGL, BLEIR.offset(l1, 1))),
        ]
    finally:
        Belex.pop_context()


def test_lx_regs() -> None:
    belex = Belex.push_context()

    try:
        l1 = L1(symbol="l1")
        assert l1.as_bleir() == BLEIR.L1_REG(l1.symbol)

        assert l1() + 0 == OffsetLX(l1(), row_id=0)
        assert (l1() + 0).as_bleir() == BLEIR.LXRegWithOffsets(
            parameter=l1.as_bleir(),
            row_id=0)

        assert l1() + (1,) == OffsetLX(l1(), row_id=1)
        assert (l1() + (1,)).as_bleir() == BLEIR.LXRegWithOffsets(
            parameter=l1.as_bleir(),
            row_id=1)

        assert l1() + (2,3) == OffsetLX(l1(), row_id=3, group_id=2)
        assert (l1() + (2,3)).as_bleir() == BLEIR.LXRegWithOffsets(
            parameter=l1.as_bleir(),
            row_id=3,
            group_id=2)

        assert l1() + (3,0,5) == OffsetLX(l1(), row_id=5, group_id=0, bank_id=3)
        assert (l1() + (3,0,5)).as_bleir() == BLEIR.LXRegWithOffsets(
            parameter=l1.as_bleir(),
            row_id=5,
            group_id=0,
            bank_id=3)

        l2 = L2(symbol="l2")

        assert l2() + 0 == OffsetLX(l2(), row_id=0)
        assert (l2() + 0).as_bleir() == BLEIR.LXRegWithOffsets(
            parameter=l2.as_bleir(),
            row_id=0)

        assert l2() + (1,) == OffsetLX(l2(), row_id=1)
        assert (l2() + (1,)).as_bleir() == BLEIR.LXRegWithOffsets(
            parameter=l2.as_bleir(),
            row_id=1)

        # -----

        l1() <= GGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, l1(), GGL())
        assert belex.instructions[-1].as_patterns() == ["LX = <SRC>", "LX = GGL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(l1, GGL)),
        ]

        l1() + 2 <= GGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, l1() + 2, GGL())
        assert belex.instructions[-1].as_patterns() == ["LX = <SRC>", "LX = GGL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(BLEIR.offset(l1, 2), GGL)),
        ]

        l1() + (3,1) <= GGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, l1() + (3,1), GGL())
        assert belex.instructions[-1].as_patterns() == ["LX = <SRC>", "LX = GGL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(BLEIR.offset(l1, (3,1)), GGL)),
        ]

        l1() + (2,1,3) <= GGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, l1() + (2,1,3), GGL())
        assert belex.instructions[-1].as_patterns() == ["LX = <SRC>", "LX = GGL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(BLEIR.offset(l1, (2,1,3)), GGL)),
        ]

        # -----

        l1() <= LGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, l1(), LGL())
        assert belex.instructions[-1].as_patterns() == ["LX = LGL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(l1, LGL)),
        ]

        l1() + 2 <= LGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, l1() + 2, LGL())
        assert belex.instructions[-1].as_patterns() == ["LX = LGL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(BLEIR.offset(l1, 2), LGL)),
        ]

        l1() + (3,1) <= LGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, l1() + (3,1), LGL())
        assert belex.instructions[-1].as_patterns() == ["LX = LGL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(BLEIR.offset(l1, (3,1)), LGL)),
        ]

        l1() + (2,1,3) <= LGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, l1() + (2,1,3), LGL())
        assert belex.instructions[-1].as_patterns() == ["LX = LGL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(BLEIR.offset(l1, (2,1,3)), LGL)),
        ]

        # -----

        l2() <= GGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, l2(), GGL())
        assert belex.instructions[-1].as_patterns() == ["LX = <SRC>", "LX = GGL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(l2, GGL)),
        ]

        l2() + 1 <= GGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, l2() + 1, GGL())
        assert belex.instructions[-1].as_patterns() == ["LX = <SRC>", "LX = GGL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(BLEIR.offset(l2, 1), GGL)),
        ]

        # -----

        l2() <= LGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, l2(), LGL())
        assert belex.instructions[-1].as_patterns() == ["LX = LGL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(l2, LGL)),
        ]

        l2() + 1 <= LGL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, l2() + 1, LGL())
        assert belex.instructions[-1].as_patterns() == ["LX = LGL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(BLEIR.assign(BLEIR.offset(l2, 1), LGL)),
        ]

    finally:
        Belex.pop_context()


def test_re_reg() -> None:
    belex = Belex.push_context()

    try:
        re_reg = RE(symbol="_INTERNAL_RE_000")
        assert re_reg.as_bleir() == \
            BLEIR.RE_REG(identifier="_INTERNAL_RE_000")

        re_reg[::] <= RL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, re_reg[::], RL())
        assert belex.instructions[-1].as_patterns() == ["SB = <SRC>", "SB = RL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(
                BLEIR.masked(SM_0XFFFF.having(
                                 symbol="_INTERNAL_SM_0XFFFF",
                                 is_lowered=False,
                                 register=None),
                             BLEIR.assign(BLEIR.SB[re_reg], RL))),
        ]

        RL[::] <= re_reg()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RL[::], re_reg())
        assert belex.instructions[-1].as_patterns() == ["RL = <SB>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(
                BLEIR.masked(SM_0XFFFF.having(
                                 symbol="_INTERNAL_SM_0XFFFF",
                                 is_lowered=False,
                                 register=None),
                             BLEIR.assign(RL, BLEIR.SB[re_reg]))),
        ]
    finally:
        Belex.pop_context()


def test_ewe_reg() -> None:
    belex = Belex.push_context()

    try:
        ewe_reg = EWE(symbol="_INTERNAL_EWE_000")
        assert ewe_reg.as_bleir() == \
            BLEIR.EWE_REG(identifier="_INTERNAL_EWE_000")

        ewe_reg[::] <= RL()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, ewe_reg[::], RL())
        assert belex.instructions[-1].as_patterns() == ["SB = <SRC>", "SB = RL"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(
                BLEIR.masked(SM_0XFFFF.having(
                                 symbol="_INTERNAL_SM_0XFFFF",
                                 is_lowered=False,
                                 register=None),
                             BLEIR.assign(BLEIR.SB[ewe_reg], RL))),
        ]

        RL[::] <= ewe_reg()
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ, RL[::], ewe_reg())
        assert belex.instructions[-1].as_patterns() == ["RL = <SB>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(
                BLEIR.masked(SM_0XFFFF.having(
                                 symbol="_INTERNAL_SM_0XFFFF",
                                 is_lowered=False,
                                 register=None),
                             BLEIR.assign(RL, BLEIR.SB[ewe_reg]))),
        ]
    finally:
        Belex.pop_context()


def test_read_write_inhibit() -> None:
    belex = Belex.push_context()

    try:
        RWINH_SET["0xFFFF"]
        assert belex.instructions[-1] == \
            SM_0XFFFF.having(read_write_inhibit=True,
                             symbol="_INTERNAL_SM_0XFFFF",
                             is_lowered=False,
                             register=None)
        assert belex.instructions[-1].as_bleir() == \
            SM_0XFFFF.having(read_write_inhibit=True,
                             symbol="_INTERNAL_SM_0XFFFF",
                             is_lowered=False,
                             register=None) \
                     .as_bleir()

        RWINH_RST["0xFFFF"]
        assert belex.instructions[-1] == \
            SM_0XFFFF.having(read_write_inhibit=False,
                             symbol="_INTERNAL_SM_0XFFFF",
                             is_lowered=False,
                             register=None)
        assert belex.instructions[-1].as_bleir() == \
            SM_0XFFFF.having(read_write_inhibit=False,
                             symbol="_INTERNAL_SM_0XFFFF",
                             is_lowered=False,
                             register=None) \
                     .as_bleir()

        lvr = belex.VR()

        # with belex.read_write_inhibit():
        #     RL["0x1111"] <= lvr()
        RWINH_SET[RL["0x1111"] <= lvr()]
        assert belex.instructions[-1] == \
            AssignOperation(ASSIGN_OP.EQ,
                            RL[SM_0X1111.having(
                                   read_write_inhibit=True,
                                   symbol="_INTERNAL_SM_0X1111",
                                   is_lowered=False,
                                   register=None)],
                            lvr())
        assert belex.instructions[-1].as_patterns() == ["RL = <SB>"]
        assert belex.instructions[-1].as_bleir() == [
            BLEIR.statement(
                BLEIR.masked(SM_0X1111.having(
                                 read_write_inhibit=True,
                                 symbol="_INTERNAL_SM_0X1111",
                                 is_lowered=False,
                                 register=None),
                             BLEIR.assign(RL, BLEIR.SB[lvr]))),
        ]
    finally:
        Belex.pop_context()


def test_gl_from_inv_rl():

    @belex_apl
    def gl_from_inv_rl(Belex):
        GL[::] <= INV_RL()

    with pytest.raises(AssertionError):
        # Only "GL = RL" is supported.
        gl_from_inv_rl()


@belex_apl
def rl_from_sb_or_inv_src(Belex, dst: VR, src: VR):
    RL[::] <= 1
    RSP16[::] <= RL()
    RSP_START_RET()
    RL[::] <= src() | ~RSP16()
    RSP_END()
    dst[::] <= RL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_rl_from_sb_or_inv_src(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    rl_from_sb_or_inv_src(dst_vp, src_vp)
    assert np.array_equal(diri.hb[dst_vp], diri.hb[src_vp])
    return dst_vp


@belex_apl
def rl_or_equals_inv_src(Belex, dst: VR, val: u16):
    RL[val] <= 1
    RL[~val] <= 0
    GGL[::] <= RL()
    RL[::] |= ~GGL()
    dst[::] <= RL()


@parameterized_belex_test
def test_rl_or_equals_inv_src(diri: DIRI) -> int:
    dst_vp = 0
    val_vp = 0xBEEF
    rl_or_equals_inv_src(dst_vp, val_vp)
    assert diri.hb[dst_vp].all()
    return dst_vp


@belex_apl
def rl_or_equals_sb_and_inv_src(Belex, dst: VR, src: VR, val: u16):
    RL[val] <= 1
    RL[~val] <= 0
    GGL[::] <= RL()
    RL[::] |= src() & ~GGL()
    dst[::] <= RL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_rl_or_equals_sb_and_inv_src(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    val_vp = 0xBEEF
    rl_or_equals_sb_and_inv_src(dst_vp, src_vp, val_vp)
    expected_value = u16_to_bool(val_vp)
    expected_value[::, 4:] |= diri.hb[src_vp, ::, 4:]
    assert np.array_equal(expected_value, diri.hb[dst_vp])
    return dst_vp


@belex_apl
def rl_and_equals_sb_and_inv_src(Belex, dst: VR, src: VR, val: u16):
    RL[val] <= 1
    RL[~val] <= 0
    GGL[::] <= RL()
    RL[::] &= src() & ~GGL()
    dst[::] <= RL()


@parameterized_belex_test
def test_rl_and_equals_sb_and_inv_src(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    val_vp = 0xBEEF
    rl_and_equals_sb_and_inv_src(dst_vp, src_vp, val_vp)
    expected_value = u16_to_bool(val_vp)
    expected_value[::, :4] = False
    expected_value[::, 4:] &= diri.hb[src_vp, ::, 4:]
    assert np.array_equal(expected_value, diri.hb[dst_vp])
    return dst_vp


@belex_apl
def rl_xor_equals_sb_and_inv_src(Belex, dst: VR, src: VR, val: u16):
    RL[val] <= 1
    RL[~val] <= 0
    GGL[::] <= RL()
    RL[::] ^= src() & ~GGL()
    dst[::] <= RL()


@parameterized_belex_test
def test_rl_xor_equals_sb_and_inv_src(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    val_vp = 0xBEEF
    rl_xor_equals_sb_and_inv_src(dst_vp, src_vp, val_vp)
    expected_value = u16_to_bool(val_vp)
    expected_value[::, 4:] ^= diri.hb[src_vp, ::, 4:]
    assert np.array_equal(expected_value, diri.hb[dst_vp])
    return dst_vp


@belex_apl
def fsel_noop(Belex):
    FSEL_NOOP()


@parameterized_belex_test
def test_fsel_noop(diri: DIRI) -> int:
    dst_vp = 0
    reset_16(dst_vp)
    fsel_noop()
    return dst_vp


@belex_apl
def sb_cond_equals_inv_src(Belex, dst: VR, src: VR):
    RL[::] <= src()
    dst[::] |= ~RL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_sb_cond_equals_inv_src(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    expected_value = (diri.hb[dst_vp] & ~diri.hb[src_vp])
    sb_cond_equals_inv_src(dst_vp, src_vp)
    assert np.array_equal(expected_value, diri.hb[dst_vp])
    return dst_vp


@belex_apl
def ggl_from_rl_and_l1(Belex, dst: VR, rl_src: VR, l1: L1, l1_src: VR, msk: Mask):
    RL[msk] <= l1_src()
    GGL[msk] <= RL()
    l1() <= GGL()
    RL[~msk] <= 0
    RL[msk] <= rl_src()
    GGL[msk] <= RL() & l1()
    dst[msk] <= GGL()
    dst[~msk] <= RL()


@hypothesis.settings(max_examples=5, deadline=None)
@given(msk_vp=Mask_strategy())
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_ggl_from_rl_and_l1(diri: DIRI, msk_vp) -> int:
    dst_vp = 0
    rl_src_vp = 1
    l1_src_vp = 2
    l1_vp = 0

    sections = list(open_belex.common.mask.Mask(f"0x{msk_vp:04X}"))

    if msk_vp == 0x0000:
        expected_ggl = np.zeros(GGL_SHAPE, dtype=bool)
    else:
        expected_ggl = np.ones(GGL_SHAPE, dtype=bool)
        for section in sections:
            expected_ggl[::, section // 4] &= diri.hb[rl_src_vp, ::, section]
            expected_ggl[::, section // 4] &= diri.hb[l1_src_vp, ::, section]

    expected_value = diri.build_vr()
    for section in sections:
        expected_value[::, section] = expected_ggl[::, section // 4]

    ggl_from_rl_and_l1(dst_vp, rl_src_vp, l1_vp, l1_src_vp, msk_vp)
    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def write_to_and_read_from_parity_row(Belex, dst: VR, src: VR, tmp: L1) -> None:
    RL[::] <= src()

    GGL[SM_0X1111 << 0] <= RL()
    tmp() <= GGL()
    GGL() <= tmp()
    dst[SM_0X1111 << 0] <= GGL()

    GGL[SM_0X1111 << 1] <= RL()
    tmp() <= GGL()
    GGL() <= tmp()
    dst[SM_0X1111 << 1] <= GGL()

    GGL[SM_0X1111 << 2] <= RL()
    tmp() <= GGL()
    GGL() <= tmp()
    dst[SM_0X1111 << 2] <= GGL()

    GGL[SM_0X1111 << 3] <= RL()
    tmp() <= GGL()
    GGL() <= tmp()
    dst[SM_0X1111 << 3] <= GGL()

    RL[::] ^= dst()
    dst[::] <= RL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_parity_addr_spill_restore(diri: DIRI) -> int:
    vmr = 0
    parity_grp, parity_row, row = belex_gal_vm_reg_to_set_ext(vmr)

    dst = 0
    src = 1

    assert diri.hb[src].any()
    assert diri.hb[dst].any()

    write_to_and_read_from_parity_row(dst, src, parity_row)
    assert not diri.hb[dst].any()

    return dst


@belex_apl
def too_many_laned_instrs(Belex):
    tmp = Belex.VR()
    with apl_commands():
        tmp[::] <= RL()
        RL[SM_0X1111 << 0] <= RSP16()
        RL[SM_0X1111 << 1] <= RSP16()
        RL[SM_0X1111 << 2] <= RSP16()
        RL[SM_0X1111 << 3] <= RSP16()


@parameterized_belex_test(interpret=False, generate_code=False)
def test_too_many_laned_instrs(diri: DIRI):
    too_many_laned_instrs()  # Should not error (prints a warning)


@seu_context
def test_apl_set_sm_reg(seu: SEULayer):
    with pytest.raises(KeyError):
        seu.sm_regs[0]
    apl_set_sm_reg(SM_REG_0, 0x0001 << 0)
    assert seu.sm_regs[0] == 0x0001 << 0

    with pytest.raises(KeyError):
        seu.sm_regs[1]
    apl_set_sm_reg(SM_REG_1, 0x0001 << 1)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1

    with pytest.raises(KeyError):
        seu.sm_regs[2]
    apl_set_sm_reg(SM_REG_2, 0x0001 << 2)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2

    with pytest.raises(KeyError):
        seu.sm_regs[3]
    apl_set_sm_reg(SM_REG_3, 0x0001 << 3)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3

    with pytest.raises(KeyError):
        seu.sm_regs[4]
    apl_set_sm_reg(SM_REG_4, 0x0001 << 4)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4

    with pytest.raises(KeyError):
        seu.sm_regs[5]
    apl_set_sm_reg(SM_REG_5, 0x0001 << 5)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5

    with pytest.raises(KeyError):
        seu.sm_regs[6]
    apl_set_sm_reg(SM_REG_6, 0x0001 << 6)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6

    with pytest.raises(KeyError):
        seu.sm_regs[7]
    apl_set_sm_reg(SM_REG_7, 0x0001 << 7)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7

    with pytest.raises(KeyError):
        seu.sm_regs[8]
    apl_set_sm_reg(SM_REG_8, 0x0001 << 8)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8

    with pytest.raises(KeyError):
        seu.sm_regs[9]
    apl_set_sm_reg(SM_REG_9, 0x0001 << 9)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    assert seu.sm_regs[9] == 0x0001 << 9

    with pytest.raises(KeyError):
        seu.sm_regs[10]
    apl_set_sm_reg(SM_REG_10, 0x0001 << 10)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    assert seu.sm_regs[9] == 0x0001 << 9
    assert seu.sm_regs[10] == 0x0001 << 10

    with pytest.raises(KeyError):
        seu.sm_regs[11]
    apl_set_sm_reg(SM_REG_11, 0x0001 << 11)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    assert seu.sm_regs[9] == 0x0001 << 9
    assert seu.sm_regs[10] == 0x0001 << 10
    assert seu.sm_regs[11] == 0x0001 << 11

    with pytest.raises(KeyError):
        seu.sm_regs[12]
    apl_set_sm_reg(SM_REG_12, 0x0001 << 12)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    assert seu.sm_regs[9] == 0x0001 << 9
    assert seu.sm_regs[10] == 0x0001 << 10
    assert seu.sm_regs[11] == 0x0001 << 11
    assert seu.sm_regs[12] == 0x0001 << 12

    with pytest.raises(KeyError):
        seu.sm_regs[13]
    apl_set_sm_reg(SM_REG_13, 0x0001 << 13)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    assert seu.sm_regs[9] == 0x0001 << 9
    assert seu.sm_regs[10] == 0x0001 << 10
    assert seu.sm_regs[11] == 0x0001 << 11
    assert seu.sm_regs[12] == 0x0001 << 12
    assert seu.sm_regs[13] == 0x0001 << 13

    with pytest.raises(KeyError):
        seu.sm_regs[14]
    apl_set_sm_reg(SM_REG_14, 0x0001 << 14)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    assert seu.sm_regs[9] == 0x0001 << 9
    assert seu.sm_regs[10] == 0x0001 << 10
    assert seu.sm_regs[11] == 0x0001 << 11
    assert seu.sm_regs[12] == 0x0001 << 12
    assert seu.sm_regs[13] == 0x0001 << 13
    assert seu.sm_regs[14] == 0x0001 << 14

    with pytest.raises(KeyError):
        seu.sm_regs[15]
    apl_set_sm_reg(SM_REG_15, 0x0001 << 15)
    assert seu.sm_regs[0] == 0x0001 << 0
    assert seu.sm_regs[1] == 0x0001 << 1
    assert seu.sm_regs[2] == 0x0001 << 2
    assert seu.sm_regs[3] == 0x0001 << 3
    assert seu.sm_regs[4] == 0x0001 << 4
    assert seu.sm_regs[5] == 0x0001 << 5
    assert seu.sm_regs[6] == 0x0001 << 6
    assert seu.sm_regs[7] == 0x0001 << 7
    assert seu.sm_regs[8] == 0x0001 << 8
    assert seu.sm_regs[9] == 0x0001 << 9
    assert seu.sm_regs[10] == 0x0001 << 10
    assert seu.sm_regs[11] == 0x0001 << 11
    assert seu.sm_regs[12] == 0x0001 << 12
    assert seu.sm_regs[13] == 0x0001 << 13
    assert seu.sm_regs[14] == 0x0001 << 14
    assert seu.sm_regs[15] == 0x0001 << 15


@seu_context
def test_apl_set_rn_reg(seu: SEULayer):
    with pytest.raises(KeyError):
        seu.rn_regs[0]
    apl_set_rn_reg(RN_REG_0, 0)
    assert seu.rn_regs[0] == 0

    with pytest.raises(KeyError):
        seu.rn_regs[1]
    apl_set_rn_reg(RN_REG_1, 1)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1

    with pytest.raises(KeyError):
        seu.rn_regs[2]
    apl_set_rn_reg(RN_REG_2, 2)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2

    with pytest.raises(KeyError):
        seu.rn_regs[3]
    apl_set_rn_reg(RN_REG_3, 3)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3

    with pytest.raises(KeyError):
        seu.rn_regs[4]
    apl_set_rn_reg(RN_REG_4, 4)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4

    with pytest.raises(KeyError):
        seu.rn_regs[5]
    apl_set_rn_reg(RN_REG_5, 5)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5

    with pytest.raises(KeyError):
        seu.rn_regs[6]
    apl_set_rn_reg(RN_REG_6, 6)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6

    with pytest.raises(KeyError):
        seu.rn_regs[7]
    apl_set_rn_reg(RN_REG_7, 7)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7

    with pytest.raises(KeyError):
        seu.rn_regs[8]
    apl_set_rn_reg(RN_REG_8, 8)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8

    with pytest.raises(KeyError):
        seu.rn_regs[9]
    apl_set_rn_reg(RN_REG_9, 9)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    assert seu.rn_regs[9] == 9

    with pytest.raises(KeyError):
        seu.rn_regs[10]
    apl_set_rn_reg(RN_REG_10, 10)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    assert seu.rn_regs[9] == 9
    assert seu.rn_regs[10] == 10

    with pytest.raises(KeyError):
        seu.rn_regs[11]
    apl_set_rn_reg(RN_REG_11, 11)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    assert seu.rn_regs[9] == 9
    assert seu.rn_regs[10] == 10
    assert seu.rn_regs[11] == 11

    with pytest.raises(KeyError):
        seu.rn_regs[12]
    apl_set_rn_reg(RN_REG_12, 12)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    assert seu.rn_regs[9] == 9
    assert seu.rn_regs[10] == 10
    assert seu.rn_regs[11] == 11
    assert seu.rn_regs[12] == 12

    with pytest.raises(KeyError):
        seu.rn_regs[13]
    apl_set_rn_reg(RN_REG_13, 13)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    assert seu.rn_regs[9] == 9
    assert seu.rn_regs[10] == 10
    assert seu.rn_regs[11] == 11
    assert seu.rn_regs[12] == 12
    assert seu.rn_regs[13] == 13

    with pytest.raises(KeyError):
        seu.rn_regs[14]
    apl_set_rn_reg(RN_REG_14, 14)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    assert seu.rn_regs[9] == 9
    assert seu.rn_regs[10] == 10
    assert seu.rn_regs[11] == 11
    assert seu.rn_regs[12] == 12
    assert seu.rn_regs[13] == 13
    assert seu.rn_regs[14] == 14

    with pytest.raises(KeyError):
        seu.rn_regs[15]
    apl_set_rn_reg(RN_REG_15, 15)
    assert seu.rn_regs[0] == 0
    assert seu.rn_regs[1] == 1
    assert seu.rn_regs[2] == 2
    assert seu.rn_regs[3] == 3
    assert seu.rn_regs[4] == 4
    assert seu.rn_regs[5] == 5
    assert seu.rn_regs[6] == 6
    assert seu.rn_regs[7] == 7
    assert seu.rn_regs[8] == 8
    assert seu.rn_regs[9] == 9
    assert seu.rn_regs[10] == 10
    assert seu.rn_regs[11] == 11
    assert seu.rn_regs[12] == 12
    assert seu.rn_regs[13] == 13
    assert seu.rn_regs[14] == 14
    assert seu.rn_regs[15] == 15


@seu_context
def test_apl_set_re_reg(seu: SEULayer):
    with pytest.raises(KeyError):
        seu.re_regs[0]
    apl_set_re_reg(RE_REG_0, 0x00FFFF << 0)
    assert seu.re_regs[0] == 0x00FFFF << 0

    with pytest.raises(KeyError):
        seu.re_regs[1]
    apl_set_re_reg(RE_REG_1, 0x00FFFF << 1)
    assert seu.re_regs[0] == 0x00FFFF << 0
    assert seu.re_regs[1] == 0x00FFFF << 1

    with pytest.raises(KeyError):
        seu.re_regs[2]
    apl_set_re_reg(RE_REG_2, 0x00FFFF << 2)
    assert seu.re_regs[0] == 0x00FFFF << 0
    assert seu.re_regs[1] == 0x00FFFF << 1
    assert seu.re_regs[2] == 0x00FFFF << 2

    with pytest.raises(KeyError):
        seu.re_regs[3]
    apl_set_re_reg(RE_REG_3, 0x00FFFF << 3)
    assert seu.re_regs[0] == 0x00FFFF << 0
    assert seu.re_regs[1] == 0x00FFFF << 1
    assert seu.re_regs[2] == 0x00FFFF << 2
    assert seu.re_regs[3] == 0x00FFFF << 3


@seu_context
def test_apl_set_ewe_reg(seu: SEULayer):
    with pytest.raises(KeyError):
        seu.ewe_regs[0]
    apl_set_ewe_reg(EWE_REG_0, 0x0FF)
    assert seu.ewe_regs[0] == 0x0FF

    with pytest.raises(KeyError):
        seu.ewe_regs[1]
    apl_set_ewe_reg(EWE_REG_1, 0x10F)
    assert seu.ewe_regs[0] == 0x0FF
    assert seu.ewe_regs[1] == 0x10F

    with pytest.raises(KeyError):
        seu.ewe_regs[2]
    apl_set_ewe_reg(EWE_REG_2, 0x234)
    assert seu.ewe_regs[0] == 0x0FF
    assert seu.ewe_regs[1] == 0x10F
    assert seu.ewe_regs[2] == 0x234

    with pytest.raises(KeyError):
        seu.ewe_regs[3]
    apl_set_ewe_reg(EWE_REG_3, 0x2FF)
    assert seu.ewe_regs[0] == 0x0FF
    assert seu.ewe_regs[1] == 0x10F
    assert seu.ewe_regs[2] == 0x234
    assert seu.ewe_regs[3] == 0x2FF


@seu_context
def test_apl_set_l1_reg(seu: SEULayer):
    with pytest.raises(KeyError):
        seu.l1_regs[0]
    apl_set_l1_reg_ext(L1_ADDR_REG_0, 1, 2, 3)
    assert seu.l1_regs[0] == (1 << 11) | (2 << 9) | 3
    apl_set_l1_reg(L1_ADDR_REG_0, 0)
    assert seu.l1_regs[0] == 0

    with pytest.raises(KeyError):
        seu.l1_regs[1]
    apl_set_l1_reg_ext(L1_ADDR_REG_1, 2, 1, 4)
    assert seu.l1_regs[1] == (2 << 11) | (1 << 9) | 4
    apl_set_l1_reg(L1_ADDR_REG_1, 16)
    assert seu.l1_regs[0] == 0
    assert seu.l1_regs[1] == 16

    with pytest.raises(KeyError):
        seu.l1_regs[2]
    apl_set_l1_reg_ext(L1_ADDR_REG_2, 0, 0, 32)
    assert seu.l1_regs[2] == (0 << 11) | (0 << 9) | 32
    apl_set_l1_reg(L1_ADDR_REG_2, 32)
    assert seu.l1_regs[0] == 0
    assert seu.l1_regs[1] == 16
    assert seu.l1_regs[2] == 32

    with pytest.raises(KeyError):
        seu.l1_regs[3]
    apl_set_l1_reg_ext(L1_ADDR_REG_3, 0, 3, 64)
    assert seu.l1_regs[3] == (0 << 11) | (3 << 9) | 64
    apl_set_l1_reg(L1_ADDR_REG_3, 48)
    assert seu.l1_regs[0] == 0
    assert seu.l1_regs[1] == 16
    assert seu.l1_regs[2] == 32
    assert seu.l1_regs[3] == 48


@seu_context
def test_apl_set_l2_reg(seu: SEULayer):
    with pytest.raises(KeyError):
        seu.l2_regs[0]
    apl_set_l2_reg(L2_ADDR_REG_0, 42)
    assert seu.l2_regs[0] == 42


@belex_apl
def cpy_vr_w_seu(Belex, dst: VR, src: VR):
    RL[::] <= src()
    dst[::] <= RL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_cpy_vr_w_seu(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1

    expected_value = deepcopy(diri.hb[src_vp])
    assert not np.array_equal(expected_value, diri.hb[dst_vp])

    apl_set_rn_reg(RN_REG_0, dst_vp)
    apl_set_rn_reg(RN_REG_1, src_vp)
    cpy_vr_w_seu(dst=RN_REG_0, src=RN_REG_1)

    assert np.array_equal(expected_value, diri.hb[dst_vp])


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_cpy_imm_16_w_seu(diri: DIRI):
    dst_vp = 0
    imm_vp = 0xBEEF

    expected_value = u16_to_bool(imm_vp)
    assert not np.array_equal(expected_value, diri.hb[dst_vp])

    apl_set_rn_reg(RN_REG_0, dst_vp)
    apl_set_sm_reg(SM_REG_0, imm_vp)
    cpy_imm_16(tgt=RN_REG_0, val=SM_REG_0)

    assert np.array_equal(expected_value, diri.hb[dst_vp])


@belex_apl
def conjoin_w_seu(Belex, dst: VR, srcs: RE):
    RL[::] <= srcs()
    dst[::] <= RL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_conjoin_w_seu(diri: DIRI):
    dst_vp = 0
    srcs_vp = 0x71F0FF

    expected_value = np.ones((NUM_PLATS_PER_APUC, NSECTIONS), dtype=bool)
    for sb in range(NSB):
        if srcs_vp & (0x000001 << sb) != 0:
            expected_value &= diri.hb[sb]

    apl_set_rn_reg(RN_REG_5, dst_vp)
    apl_set_re_reg(RE_REG_2, srcs_vp)
    conjoin_w_seu(dst=RN_REG_5, srcs=RE_REG_2)

    assert np.array_equal(expected_value, diri.hb[dst_vp])


@belex_apl
def broadcast_w_seu(Belex, dsts: EWE, src: VR):
    RL[::] <= src()
    dsts[::] <= RL()


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_broadcast_w_seu(diri: DIRI):
    dsts_vp = 0x123
    src_vp = 3

    offset = 8 * (dsts_vp >> 8)
    expected_value = deepcopy(diri.hb[src_vp])

    apl_set_ewe_reg(EWE_REG_2, dsts_vp)
    apl_set_rn_reg(RN_REG_11, src_vp)
    broadcast_w_seu(dsts=EWE_REG_2, src=RN_REG_11)

    for sb in range(8):
        if dsts_vp & (0x01 << sb) != 0:  # the offset is wrong !!!
            assert np.array_equal(expected_value, diri.hb[offset + sb])


GSI_L2_CTL_ROW_ADDR_BIT_IDX_BITS: int = 4


def belex_gal_encode_l2_addr(byte_idx: int, bit_idx: int) -> int:
    return ((byte_idx << GSI_L2_CTL_ROW_ADDR_BIT_IDX_BITS) | bit_idx)


def bank_group_row_to_addr(bank_id: int, group_id: int, row_id: int) -> int:
    return (bank_id << 11) \
        | (group_id << 9) \
        | row_id


@c_caller
def belex_l2dma_l2_ready_rst_all(diri: DIRI) -> None:
    pass


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_out_in_w_seu(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1

    vm_reg = 0
    l1_parity_grp, l1_parity_row, l1_grp_row = \
        belex_gal_vm_reg_to_set_ext(vm_reg)

    l1_parity_mask = store_16_parity_mask(l1_parity_grp)

    apl_set_l1_reg(L1_ADDR_REG_0, l1_grp_row)
    apl_set_l1_reg(L1_ADDR_REG_1, l1_parity_row)
    apl_set_sm_reg(SM_REG_0, l1_parity_mask)
    apl_set_rn_reg(RN_REG_0, src_vp)
    store_16_t0(dst=L1_ADDR_REG_0,
                parity_dst=L1_ADDR_REG_1,
                parity_mask=SM_REG_0,
                src=RN_REG_0)

    expected_value = deepcopy(diri.hb[src_vp])

    belex_l2dma_l2_ready_rst_all()
    for l1_bank_id in range(4):
        l1_grp = 0
        l2_start_byte = l1_bank_id * 2
        for i in range(2):
            l2_addr = belex_gal_encode_l2_addr(l2_start_byte + i, 0)
            src_addr = bank_group_row_to_addr(l1_bank_id, l1_grp, l1_grp_row)
            parity_src_addr = \
                bank_group_row_to_addr(l1_bank_id, l1_parity_grp, l1_parity_row)

            apl_set_l2_reg(L2_ADDR_REG_0, l2_addr)
            apl_set_l1_reg(L1_ADDR_REG_0, src_addr)
            apl_set_l1_reg(L1_ADDR_REG_1, parity_src_addr)
            copy_l1_to_l2_byte(dst=L2_ADDR_REG_0,
                               src=L1_ADDR_REG_0,
                               parity_src=L1_ADDR_REG_1)

            l1_grp += 2

    l2_end()

    # Use a different VMR to avoid data artifacts in the case of a bug that does
    # not update every bit.

    vm_reg = 1

    l1_parity_grp, l1_parity_row, l1_grp_row = \
        belex_gal_vm_reg_to_set_ext(vm_reg)

    belex_l2dma_l2_ready_rst_all()
    for l1_bank_id in range(4):
        l1_grp = 0
        l2_start_byte = l1_bank_id * 2
        for i in range(2):
            l2_addr = belex_gal_encode_l2_addr(l2_start_byte + i, 0)
            dst_addr = bank_group_row_to_addr(l1_bank_id, l1_grp, l1_grp_row)
            parity_dst_addr = \
                bank_group_row_to_addr(l1_bank_id, l1_parity_grp, l1_parity_row)

            apl_set_l1_reg(L1_ADDR_REG_0, dst_addr)
            apl_set_l1_reg(L1_ADDR_REG_1, parity_dst_addr)
            apl_set_l2_reg(L2_ADDR_REG_0, l2_addr)
            copy_l2_to_l1_byte(dst=L1_ADDR_REG_0,
                               parity_dst=L1_ADDR_REG_1,
                               src=L2_ADDR_REG_0)

            l1_grp += 2

    l2_end()

    vm_reg = 1
    l1_parity_grp, l1_parity_row, l1_grp_row = \
        belex_gal_vm_reg_to_set_ext(vm_reg)
    l1_parity_mask = load_16_parity_mask(l1_parity_grp)

    apl_set_rn_reg(RN_REG_0, dst_vp)
    apl_set_l1_reg(L1_ADDR_REG_0, l1_grp_row)
    apl_set_l1_reg(L1_ADDR_REG_1, l1_parity_row)
    apl_set_sm_reg(SM_REG_0, l1_parity_mask)
    load_16_t0(dst=RN_REG_0,
               src=L1_ADDR_REG_0,
               parity_src=L1_ADDR_REG_1,
               parity_mask=SM_REG_0)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_mixing_automatic_and_manual_resource_management(diri: DIRI) -> int:
    # It is not recommended to mix automatic and manual resource management,
    # but such heterogeneous applications must be (at least partially)
    # supported.

    dst_vp = 0
    src_vp = 1
    imm_vp = 0x1234

    expected_value = u16_to_bool(imm_vp)

    # reserve RN_REG_1 and SM_REG_0 so they will not be used by cpy_16
    apl_set_rn_reg(RN_REG_1, src_vp)
    apl_set_sm_reg(SM_REG_0, imm_vp)
    cpy_imm_16(tgt=RN_REG_1, val=SM_REG_0)

    # cpy_16 should use RN_REG_0 and RN_REG_2, but not RN_REG_1
    cpy_16(dst_vp, src_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


def append_01(src: VR):
    RL[::] <= src()
    with appendable_commands():
        RL[::] <= RSP16()


def append_02(dst: VR):
    with apl_commands():
        dst[::] <= RL()


@belex_apl
def append_lanes(Belex, dst: VR, src: VR):
    with append_commands():
        append_01(src)
        append_02(dst)


@parameterized_belex_test(repeatably_randomize_half_bank=True,
                          generate_code=False,
                          interpret=False)
def test_append_lanes(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    expected_value = deepcopy(diri.hb[src_vp])
    fragment_caller_call = append_lanes(dst_vp, src_vp)
    assert np.array_equal(expected_value, diri.hb[src_vp])
    fragment = fragment_caller_call.fragment
    assert len(fragment.operations) == 2
    assert isinstance(fragment.operations[0], BLEIR.STATEMENT)
    assert str(fragment.operations[0]) == "_INTERNAL_SM_0XFFFF: RL = SB[src];"
    assert isinstance(fragment.operations[1], BLEIR.MultiStatement)
    assert str(fragment.operations[1]) == "{" + " ".join([
        "_INTERNAL_SM_0XFFFF: RL = RSP16;",
        "_INTERNAL_SM_0XFFFF: SB[dst] = RL;",
    ]) + "}"
    return dst_vp


def fn_one_part(dst: VR, src: VR):
    RL[::] <= src()
    with appendable_commands():
        dst[::] <= RL()


@belex_apl
def appendable_lanes_one_part(Belex, dst: VR, src: VR):
    with append_commands():
        # Execute the appendable_commands() as an apl_commands()
        fn_one_part(dst, src)


@parameterized_belex_test(repeatably_randomize_half_bank=True,
                          generate_code=False,
                          interpret=False)
def test_appendable_lanes_one_part(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    expected_value = deepcopy(diri.hb[src_vp])
    fragment_caller_call = appendable_lanes_one_part(dst_vp, src_vp)
    assert np.array_equal(expected_value, diri.hb[src_vp])
    fragment = fragment_caller_call.fragment
    assert len(fragment.operations) == 2
    assert isinstance(fragment.operations[0], BLEIR.STATEMENT)
    assert str(fragment.operations[0]) == "_INTERNAL_SM_0XFFFF: RL = SB[src];"
    assert isinstance(fragment.operations[1], BLEIR.MultiStatement)
    assert str(fragment.operations[1]) == "{" + " ".join([
        "_INTERNAL_SM_0XFFFF: SB[dst] = RL;",
    ]) + "}"
    return dst_vp


def fn_no_parts_01(src: VR):
    with apl_commands():
        RL[::] <= src()


def fn_no_parts_02(dst: VR):
    with apl_commands():
        dst[::] <= RL()


@belex_apl
def appendable_lanes_no_parts(Belex, dst: VR, src: VR):
    with append_commands():
        fn_no_parts_01(src)
        fn_no_parts_02(dst)


@parameterized_belex_test(repeatably_randomize_half_bank=True,
                          generate_code=False,
                          interpret=False)
def test_appendable_lanes_no_parts(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    expected_value = deepcopy(diri.hb[src_vp])
    fragment_caller_call = appendable_lanes_no_parts(dst_vp, src_vp)
    assert np.array_equal(expected_value, diri.hb[src_vp])
    fragment = fragment_caller_call.fragment
    assert len(fragment.operations) == 2
    assert isinstance(fragment.operations[0], BLEIR.MultiStatement)
    assert str(fragment.operations[0]) == "{_INTERNAL_SM_0XFFFF: RL = SB[src];}"
    assert isinstance(fragment.operations[1], BLEIR.MultiStatement)
    assert str(fragment.operations[1]) == "{_INTERNAL_SM_0XFFFF: SB[dst] = RL;}"
    return dst_vp


@belex_apl
def empty_appendable_lanes(Belex):
    with append_commands():
        pass


@parameterized_belex_test(repeatably_randomize_half_bank=True,
                          generate_code=False,
                          interpret=False)
def test_empty_appendable_lanes(diri: DIRI):
    fragment_caller_call = empty_appendable_lanes()  # should not raise an error
    fragment = fragment_caller_call.fragment
    assert len(fragment.operations) == 0


@belex_apl
def cpy_w_appendable_commands(Belex, dst: VR, src: VR) -> None:
    RL[::] <= src()
    with appendable_commands():
        dst[::] <= RL()


@parameterized_belex_test(repeatably_randomize_half_bank=True,
                          generate_code=False,
                          interpret=False)
def test_cpy_w_appendable_commands(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    expected_value = deepcopy(diri.hb[src_vp])
    fragment_caller_call = cpy_w_appendable_commands(dst_vp, src_vp)
    fragment = fragment_caller_call.fragment
    assert np.array_equal(expected_value, diri.hb[dst_vp])
    assert len(fragment.operations) == 2
    return dst_vp


def fn_ends_appendable_01(src: VR):
    RL[::] <= RSP16()
    with appendable_commands():
        RL[::] <= src()


def fn_ends_appendable_02(dst: VR, src: VR):
    with appendable_commands():
        dst[::] <= RL()


@belex_apl
def ends_appendable(Belex, dst: VR, src: VR):
    fn_ends_appendable_01(src)
    fn_ends_appendable_02(dst, src)


@parameterized_belex_test(repeatably_randomize_half_bank=True,
                          generate_code=False,
                          interpret=False)
def test_fn_ends_appendable(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    expected_value = deepcopy(diri.hb[src_vp])
    ends_appendable(dst_vp, src_vp)
    assert np.array_equal(expected_value, diri.hb[dst_vp])
    return dst_vp


@belex_apl
def fill_gl_w_rwinh(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= 1
    GL[msk] <= RL()
    dst[::] <= GL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_gl_w_rwinh(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = diri.build_vr()
    expected_value[0::2, ::] = True

    diri.hb[mrk_vp, ::2, ::] = True

    fill_gl_w_rwinh(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_ggl_w_rwinh(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= 1
    GGL[::] <= RL()
    dst[::] <= GGL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_ggl_w_rwinh(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x000F)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_ggl_w_rwinh(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rsp16_w_rwinh(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 1
    RSP16[::] <= RL()
    RSP_START_RET()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    dst[msk] <= RSP16()
    RSP_END()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rsp16_w_rwinh(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rsp16_w_rwinh(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def interleave_existing_w_rwinh(Belex, dst: VR, tmp: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    tmp[::] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= 1
    tmp[msk] <= RL()
    RWINH_RST[msk]

    RL[::] <= mrk()
    mrk[::] <= WRL()

    RWINH_SET[RL[::] <= mrk()]
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_RST[::]

    RL[::] <= tmp()
    dst[::] |= WRL()


@parameterized_belex_test
def test_interleave_existing_w_rwinh(diri: DIRI):
    dst_vp = 0
    tmp_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[0::2, ::] = u16_to_bool(0xABCD, num_plats=1)

    diri.hb[dst_vp, ::, ::] = u16_to_bool(0xABCD)
    diri.hb[mrk_vp, ::2, ::] = True

    interleave_existing_w_rwinh(dst_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def set_rst_rwinh_no_lane(Belex, dst: VR, mrk: VR):
    RL[::] <= 0
    RWINH_SET[RL["0xBEEF"] <= mrk()]
    RL["0xBE00"] <= 0
    RL["0x00EF"] <= 1
    RWINH_RST["0xBE00"]
    dst[::] <= RL()
    RWINH_RST["0x00EF"]


@parameterized_belex_test(features={
    "auto-merge-commands": False,
})
def test_set_rst_rwinh_no_lane(diri: DIRI):
    dst_vp = 0
    mrk_vp = 2

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    set_rst_rwinh_no_lane(dst_vp, mrk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def set_rst_rwinh_w_lane(Belex, dst: VR, mrk: VR, msk: Mask):
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL["0xBE00"] <= 0
    RL["0x00EF"] <= 1
    RWINH_RST["0xB000"]
    RWINH_RST["0x0E00"]
    dst[::] <= RL()
    RWINH_RST["0xFFFF"]


@parameterized_belex_test
def test_set_rst_rwinh_w_lane(diri: DIRI):
    dst_vp = 0
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    set_rst_rwinh_w_lane(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rsp16_w_rwinh_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 1
    RSP16[::] <= RL()
    RSP_START_RET()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    dst[::] <= RSP16()
    RSP_END()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rsp16_w_rwinh_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x4110)
    expected_value[0::2, ::] = True

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rsp16_w_rwinh_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_inv_rsp16_w_rwinh(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    dst[msk] <= INV_RSP16()
    RSP_END()  # <-- Why is this necessary? It seems the WRITE from INV_RSP16
               # ^-- is unaffected by RWINH unless RSP_END is used.
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_inv_rsp16_w_rwinh(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_inv_rsp16_w_rwinh(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_inv_rsp16_w_rwinh_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    dst[::] <= INV_RSP16()
    RSP_END()  # <-- Why is this necessary? It seems the WRITE from INV_RSP16
               # ^-- is unaffected by RWINH unless RSP_END is used.
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_inv_rsp16_w_rwinh_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x4110)
    expected_value[0::2, ::] = True

    diri.hb[mrk_vp, ::2, ::] = True

    fill_inv_rsp16_w_rwinh_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_sb_cond_eq_rsp16_w_rwinh(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[msk] <= 1
    RSP16[msk] <= RL()
    RSP_START_RET()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    dst[msk] |= RSP16()
    RSP_END()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_sb_cond_eq_rsp16_w_rwinh(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_sb_cond_eq_rsp16_w_rwinh(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_sb_cond_eq_inv_rsp16_w_rwinh(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[~msk] <= 0
    RL[msk] <= 1
    RSP16[msk] <= RL()
    RSP16[~msk] <= RL()
    RSP_START_RET()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    dst[::] |= RSP16()
    RSP_END()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_sb_cond_eq_inv_rsp16_w_rwinh(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_sb_cond_eq_inv_rsp16_w_rwinh(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_rl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= 1
    dst[msk] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_rl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_rl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_rl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= 1
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_rl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = diri.build_vr()
    expected_value[0::2, ::] = True
    expected_value[1::2, ::] = u16_to_bool(0x4110, num_plats=1)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_rl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_rl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= 1
    RL[~msk] <= 0
    dst[msk] <= INV_RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_rl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_rl(dst_vp, mrk_vp, msk_vp)

    assert not diri.hb[dst_vp].any()

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_rl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= 1
    RL[~msk] <= 0
    dst[::] <= INV_RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_rl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x4110)

    diri.hb[mrk_vp, 0::2, ::] = True
    diri.hb[mrk_vp, 1::2, ::] = False

    fill_rwinh_w_inv_rl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_nrl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= 1
    dst[msk] <= NRL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_nrl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x3CCE)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_nrl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_nrl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= 1
    dst[::] <= NRL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_nrl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x3CCE)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_nrl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_nrl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= 1
    dst[msk] <= INV_NRL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_nrl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x8221)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_nrl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_nrl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= 1
    dst[::] <= INV_NRL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_nrl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0001)
    expected_value[1::2, ::] = u16_to_bool(0x4110, num_plats=1)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_nrl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_erl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= 1
    dst[::] <= ERL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_erl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x4110)

    # With ERL, the final plat of each half-bank is 0x0000
    for half_bank in range(16):
        expected_value[(half_bank + 1) * 2048 - 1, ::] = False

    diri.hb[mrk_vp, 0::2, ::] = True
    diri.hb[mrk_vp, 1::2, ::] = False

    fill_rwinh_w_erl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_erl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= 1
    dst[::] <= ERL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_erl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x4110)

    # With ERL, the final plat of each half-bank is 0x0000
    for half_bank in range(16):
        expected_value[(half_bank + 1) * 2048 - 1, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_erl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_erl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= 1
    dst[::] <= INV_ERL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_erl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    # With INV_ERL, the final plat of each half-bank is 0x4110
    for half_bank in range(16):
        expected_value[(half_bank + 1) * 2048 - 1, ::] = \
            u16_to_bool(0x4110, num_plats=1)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_erl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_erl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= 1
    dst[::] <= INV_ERL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_erl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    # With INV_ERL, the final plat of each half-bank is 0xFFFF
    for half_bank in range(16):
        expected_value[(half_bank + 1) * 2048 - 1, ::] = \
            u16_to_bool(0x4110, num_plats=1)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_erl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_wrl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= 1
    dst[msk] <= WRL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_wrl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_wrl(dst_vp, mrk_vp, msk_vp)

    assert not diri.hb[dst_vp].any()

    return dst_vp


@belex_apl
def fill_rwinh_w_wrl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= 1
    dst[::] <= WRL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_wrl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x4110)

    # With WRL, the first plat of each half-bank is 0x0000
    for half_bank in range(16):
        expected_value[half_bank * 2048, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_wrl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_wrl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= 1
    dst[msk] <= INV_WRL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_wrl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_wrl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_wrl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= 1
    dst[::] <= INV_WRL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_wrl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    # With INV_WRL, the first plat of each half-bank is 0xFFFF
    for half_bank in range(16):
        expected_value[half_bank * 2048, ::] = True

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_wrl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_srl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= 1
    dst[msk] <= SRL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_srl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x1E67)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_srl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_srl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= 1
    dst[::] <= SRL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_srl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x7FFF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_srl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_srl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= 1
    dst[msk] <= INV_SRL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_srl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xA088)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_srl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_srl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= 1
    dst[::] <= INV_SRL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_srl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x8000)
    expected_value[1::2, ::] = u16_to_bool(0x4110, num_plats=1)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_srl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_gl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 1
    GL[::] <= RL()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= GL()
    dst[msk] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_gl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_gl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_gl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 1
    GL[::] <= RL()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_gl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x4110)
    expected_value[0::2, ::] = True

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_gl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_gl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 1
    GL[::] <= RL()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= INV_GL()
    dst[msk] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_gl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0000)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_gl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_gl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 1
    GL[::] <= RL()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= INV_GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_gl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0000)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_gl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_ggl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 1
    GGL[::] <= RL()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= GGL()
    dst[msk] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_ggl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_ggl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_ggl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 1
    GGL[::] <= RL()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_ggl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x4110)
    expected_value[0::2, ::] = True

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_ggl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_ggl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 1
    GGL[::] <= RL()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= INV_GGL()
    dst[msk] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_ggl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0000)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_ggl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_ggl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 1
    GGL[::] <= RL()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= INV_GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_ggl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0000)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_ggl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_rsp16(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 1
    RSP16[msk] <= RL()
    RSP_START_RET()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= RSP16()
    RSP_END()
    dst[msk] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_rsp16(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_rsp16(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_rsp16_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 1
    RSP16[msk] <= RL()
    RSP_START_RET()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_rsp16_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_rsp16_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_rsp16(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 1
    RSP16[msk] <= RL()
    RSP_START_RET()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= INV_RSP16()
    RSP_END()
    dst[msk] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_rsp16(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0000)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_rsp16(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_rsp16_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 1
    RSP16[msk] <= RL()
    RSP_START_RET()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= INV_RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_rsp16_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x4110)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_rsp16_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_half_rwinh_w_rl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= 1
    dst[msk] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_half_rwinh_w_rl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x00EF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, :8] = True

    fill_half_rwinh_w_rl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_half_rwinh_w_rl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= 1
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_half_rwinh_w_rl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x41FF)
    expected_value[1::2, ::] = u16_to_bool(0x4110, num_plats=1)

    diri.hb[mrk_vp, ::2, :8] = True

    fill_half_rwinh_w_rl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_half_rwinh_w_inv_rl(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= 1
    dst[msk] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_half_rwinh_w_inv_rl(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x00EF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, :8] = True

    fill_half_rwinh_w_inv_rl(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_half_rwinh_w_inv_rl_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask) -> None:
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= 1
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_half_rwinh_w_inv_rl_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x41FF)
    expected_value[1::2, ::] = u16_to_bool(0x4110, num_plats=1)

    diri.hb[mrk_vp, ::2, :8] = True

    fill_half_rwinh_w_inv_rl_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp

#  ___ _             _____ _____
# | _ \ |     ___   / / __| _ ) \
# |   / |__  |___| < <\__ \ _ \> >
# |_|_\____| |___|  \_\___/___/_/

@belex_apl
def fill_rwinh_w_sb(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= src()
    dst[msk] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_sb(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = True
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_sb(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_sb_w_fsrp(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_sb_w_fsrp(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xFFFF)
    expected_value[1::2, ::] = u16_to_bool(0x4110, num_plats=1)

    diri.hb[src_vp, ::, ::] = True
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_sb_w_fsrp(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp

#  ___ _            /\/|_____ _____
# | _ \ |     ___  |/\// / __| _ ) \
# |   / |__  |___|    < <\__ \ _ \> >
# |_|_\____| |___|     \_\___/___/_/

@belex_apl
def fill_rwinh_w_inv_sb(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= ~src()
    dst[msk] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_sb(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0000)

    diri.hb[src_vp, ::, ::] = True
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_sb(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_inv_sb_w_fsrp(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= ~src()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_inv_sb_w_fsrp(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0000)

    diri.hb[src_vp, ::, ::] = True
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_inv_sb_w_fsrp(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp

#  ___ _             _____ _____    __        _____ ___  _____
# | _ \ |     ___   / / __| _ ) \  / _|___   / / __| _ \/ __\ \
# |   / |__  |___| < <\__ \ _ \> > > _|_ _| < <\__ \   / (__ > >
# |_|_\____| |___|  \_\___/___/_/  \_____|   \_\___/_|_\\___/_/

@belex_apl
def fill_rwinh_w_sb_and_rsp16(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[msk] <= 1
    RSP16[msk] <= RL()
    RSP_START_RET()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= src() & RSP16()
    RSP_END()
    dst[msk] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_sb_and_rsp16(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xAACD)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_sb_and_rsp16(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_sb_and_rsp16_w_fsrp(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[msk] <= 1
    RSP16[msk] <= RL()
    RSP_START_RET()
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src() & RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_sb_and_rsp16_w_fsrp(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xAACD)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_sb_and_rsp16_w_fsrp(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_sb_and_inv_rsp16(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[msk] <= src() & INV_RSP16()
    dst[msk] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_sb_and_inv_rsp16(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xAACD)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_sb_and_inv_rsp16(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rwinh_w_sb_and_inv_rsp16_w_fsrp(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src() & INV_RSP16()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rwinh_w_sb_and_inv_rsp16_w_fsrp(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rwinh_w_sb_and_inv_rsp16_w_fsrp(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_sb_w_gl_w_rwinh(Belex, dst: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    dst[~msk] <= RL()

    RL[msk] <= 1
    GL[msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    dst[msk] <= GL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_sb_w_gl_w_rwinh(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_sb_w_gl_w_rwinh(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_sb_w_gl_w_rwinh_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask):
    RL[::] <= 1
    GL[::] <= RL()

    RL[~msk] <= 0
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    dst[::] <= GL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_sb_w_gl_w_rwinh_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xFFFF)
    expected_value[1::2, ::] = u16_to_bool(0x4110, num_plats=1)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_sb_w_gl_w_rwinh_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_sb_w_inv_gl_w_rwinh(Belex, dst: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    dst[~msk] <= RL()

    RL[msk] <= 1
    GL[msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    dst[msk] <= INV_GL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_sb_w_inv_gl_w_rwinh(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0000)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_sb_w_inv_gl_w_rwinh(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_sb_w_inv_gl_w_rwinh_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask):
    RL[::] <= 1
    GL[::] <= RL()

    RL[~msk] <= 0
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    dst[::] <= INV_GL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_sb_w_inv_gl_w_rwinh_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0000)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_sb_w_inv_gl_w_rwinh_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_sb_w_ggl_w_rwinh(Belex, dst: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    dst[~msk] <= RL()

    RL[msk] <= 1
    GGL[msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    dst[msk] <= GGL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_sb_w_ggl_w_rwinh(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[mrk_vp, ::2, ::] = True

    fill_sb_w_ggl_w_rwinh(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_sb_w_ggl_w_rwinh_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask):
    RL[::] <= 1
    GGL[::] <= RL()

    RL[~msk] <= 0
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    dst[::] <= GGL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_sb_w_ggl_w_rwinh_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xFFFF)
    expected_value[1::2, ::] = u16_to_bool(0x4110, num_plats=1)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_sb_w_ggl_w_rwinh_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_sb_w_inv_ggl_w_rwinh(Belex, dst: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    dst[~msk] <= RL()

    RL[msk] <= 1
    GGL[msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    dst[msk] <= INV_GGL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_sb_w_inv_ggl_w_rwinh(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0000)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_sb_w_inv_ggl_w_rwinh(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_sb_w_inv_ggl_w_rwinh_w_fsrp(Belex, dst: VR, mrk: VR, msk: Mask):
    RL[::] <= 1
    GGL[::] <= RL()

    RL[~msk] <= 0
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    dst[::] <= INV_GGL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_sb_w_inv_ggl_w_rwinh_w_fsrp(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0000)

    diri.hb[mrk_vp, ::2, ::] = True

    fill_sb_w_inv_ggl_w_rwinh_w_fsrp(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_rl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= RL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_rl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_rl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_inv_rl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= INV_RL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_inv_rl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBFEF)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_inv_rl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_nrl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= NRL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_nrl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBFCF)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_nrl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_inv_nrl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= INV_NRL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_inv_nrl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABED)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_inv_nrl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_erl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= ERL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_erl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_erl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_inv_erl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= INV_ERL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_inv_erl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBFEF)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_inv_erl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_wrl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= WRL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_wrl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_wrl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp

# FIXME: This test is broken. It passed at one time but it seems that happened
# because the state of the APU was changed with other, unintentional RWINH
# settings that I have had difficulty reproducing.

# @belex_apl
# def fill_rl_or_eq_inv_wrl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
#     RL[~msk] <= 0
#     RL[msk] <= 1
#     RL[::] <= RL()
#     dst[~msk] <= RL()

#     RL[msk] <= mrk()
#     RWINH_SET[msk]
#     RL[::] <= src()
#     RL[msk] |= INV_WRL()
#     dst[::] <= RL()
#     RWINH_RST[msk]


# @parameterized_belex_test
# def test_fill_rl_or_eq_inv_wrl_w_rwinh(diri: DIRI):
#     dst_vp = 0
#     src_vp = 1
#     mrk_vp = 2
#     msk_vp = 0xBEEF

#     expected_value = u16_to_bool(0xBFEF)
#     expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

#     diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
#     diri.hb[mrk_vp, ::2, ::] = True

#     fill_rl_or_eq_inv_wrl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

#     assert np.array_equal(expected_value, diri.hb[dst_vp])

#     return dst_vp


@belex_apl
def fill_rl_or_eq_srl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= SRL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_srl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBFEF)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_srl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_inv_srl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= INV_SRL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_inv_srl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_inv_srl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_gl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[msk] <= 1
    GL[msk] <= RL()

    RL[~msk] <= 0
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_gl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBFEF)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_gl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_inv_gl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[msk] <= 1
    GL[msk] <= RL()

    RL[~msk] <= 0
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= INV_GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_inv_gl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_inv_gl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_ggl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    GGL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_ggl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCF)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_ggl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_inv_ggl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    GGL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= INV_GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_inv_ggl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBFED)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_inv_ggl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_rsp16_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RSP16[::] <= RL()
    RSP_START_RET()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_rsp16_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBFEF)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)

    diri.hb[mrk_vp, 0::2, ::] = True
    diri.hb[mrk_vp, 1::2, ::] = False

    fill_rl_or_eq_rsp16_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_inv_rsp16_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RSP16[::] <= RL()
    RSP_START_RET()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] |= INV_RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_inv_rsp16_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)

    diri.hb[mrk_vp, 0::2, ::] = True
    diri.hb[mrk_vp, 1::2, ::] = False

    fill_rl_or_eq_inv_rsp16_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_rl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] &= RL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_rl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_rl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_inv_rl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] &= INV_RL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_inv_rl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0100)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_inv_rl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp

# FIXME: This test is broken. It passed at one time but it seems that happened
# because the state of the APU was changed with other, unintentional RWINH
# settings that I have had difficulty reproducing.

# @belex_apl
# def fill_rl_and_eq_nrl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
#     RL[~msk] <= 0
#     RL[msk] <= 1
#     dst[~msk] <= RL()

#     RL[msk] <= mrk()
#     RWINH_SET[msk]
#     RL[::] <= src()
#     RL[msk] &= NRL()
#     dst[::] <= RL()
#     RWINH_RST[msk]


# @parameterized_belex_test
# def test_fill_rl_and_eq_nrl_w_rwinh(diri: DIRI):
#     dst_vp = 0
#     src_vp = 1
#     mrk_vp = 2
#     msk_vp = 0xBEEF

#     expected_value = u16_to_bool(0x0388)
#     expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

#     diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
#     diri.hb[mrk_vp, ::2, ::] = True

#     fill_rl_and_eq_nrl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

#     assert np.array_equal(expected_value, diri.hb[dst_vp])

#     return dst_vp


@belex_apl
def fill_rl_and_eq_inv_nrl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] &= INV_NRL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_inv_nrl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xA945)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_inv_nrl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_erl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] &= ERL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_erl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0100)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_erl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_inv_erl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] &= INV_ERL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_inv_erl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_inv_erl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_wrl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] &= WRL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_wrl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0100)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_wrl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_inv_wrl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] &= INV_WRL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_inv_wrl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_inv_wrl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_srl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] &= SRL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_srl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x01C4)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_srl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp

# FIXME: This test is broken. It passed at one time but it seems that happened
# because the state of the APU was changed with other, unintentional RWINH
# settings that I have had difficulty reproducing.

# @belex_apl
# def fill_rl_and_eq_inv_srl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
#     RL[~msk] <= 0
#     RL[msk] <= 1
#     RL[::] <= RL()
#     dst[~msk] <= RL()

#     RL[msk] <= mrk()
#     RWINH_SET[msk]
#     RL[::] <= src()
#     RL[msk] &= INV_SRL()
#     dst[::] <= RL()
#     RWINH_RST[msk]


# @parameterized_belex_test
# def test_fill_rl_and_eq_inv_srl_w_rwinh(diri: DIRI):
#     dst_vp = 0
#     src_vp = 1
#     mrk_vp = 2
#     msk_vp = 0xBEEF

#     expected_value = u16_to_bool(0xAB09)
#     expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

#     diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
#     diri.hb[mrk_vp, ::2, ::] = True

#     fill_rl_and_eq_inv_srl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

#     assert np.array_equal(expected_value, diri.hb[dst_vp])

#     return dst_vp


@belex_apl
def fill_rl_and_eq_gl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[msk] <= 1
    GL[msk] <= RL()

    RL[~msk] <= 0
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] &= GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_gl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_gl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_inv_gl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[msk] <= 1
    GL[msk] <= RL()

    RL[~msk] <= 0
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] &= INV_GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_inv_gl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0100)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_inv_gl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_ggl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    GGL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] &= GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_ggl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x010D)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_ggl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_inv_ggl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    GGL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] &= INV_GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_inv_ggl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABC0)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_inv_ggl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_rsp16_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RSP16[::] <= RL()
    RSP_START_RET()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] &= RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_rsp16_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_rsp16_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_inv_rsp16_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RSP16[::] <= RL()
    RSP_START_RET()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] &= INV_RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_inv_rsp16_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0100)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_inv_rsp16_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_rl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] ^= RL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_rl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0100)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_rl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_inv_rl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] ^= INV_RL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_inv_rl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBFEF)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_inv_rl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp

# FIXME: This test is broken. It passed at one time but it seems that happened
# because the state of the APU was changed with other, unintentional RWINH
# settings that I have had difficulty reproducing.

# @belex_apl
# def fill_rl_xor_eq_nrl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
#     RL[~msk] <= 0
#     RL[msk] <= 1
#     dst[~msk] <= RL()

#     RL[msk] <= mrk()
#     RWINH_SET[msk]
#     RL[::] <= src()
#     RL[msk] ^= NRL()
#     dst[::] <= RL()
#     RWINH_RST[msk]


# @parameterized_belex_test
# def test_fill_rl_xor_eq_nrl_w_rwinh(diri: DIRI):
#     dst_vp = 0
#     src_vp = 1
#     mrk_vp = 2
#     msk_vp = 0xBEEF

#     expected_value = u16_to_bool(0xBD47)
#     expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

#     diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
#     diri.hb[mrk_vp, ::2, ::] = True

#     fill_rl_xor_eq_nrl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

#     assert np.array_equal(expected_value, diri.hb[dst_vp])

#     return dst_vp


@belex_apl
def fill_rl_xor_eq_inv_nrl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] ^= INV_NRL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_inv_nrl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x03A8)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_inv_nrl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_erl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] ^= ERL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_erl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_erl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_inv_erl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] ^= INV_ERL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_inv_erl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x1522)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_inv_erl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_wrl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] ^= WRL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_wrl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_wrl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp

# FIXME: This test is broken. It passed at one time but it seems that happened
# because the state of the APU was changed with other, unintentional RWINH
# settings that I have had difficulty reproducing.

# @belex_apl
# def fill_rl_xor_eq_inv_wrl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
#     RL[~msk] <= 0
#     RL[msk] <= 1
#     RL[::] <= RL()
#     dst[~msk] <= RL()

#     RL[msk] <= mrk()
#     RWINH_SET[msk]
#     RL[::] <= src()
#     RL[msk] ^= INV_WRL()
#     dst[::] <= RL()
#     RWINH_RST[msk]


# @parameterized_belex_test
# def test_fill_rl_xor_eq_inv_wrl_w_rwinh(diri: DIRI):
#     dst_vp = 0
#     src_vp = 1
#     mrk_vp = 2
#     msk_vp = 0xBEEF

#     expected_value = u16_to_bool(0x1522)
#     expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

#     diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
#     diri.hb[mrk_vp, ::2, ::] = True

#     fill_rl_xor_eq_inv_wrl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

#     assert np.array_equal(expected_value, diri.hb[dst_vp])

#     return dst_vp


@belex_apl
def fill_rl_xor_eq_srl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] ^= SRL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_srl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBF2B)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_srl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_inv_srl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] ^= INV_SRL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_inv_srl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x01C4)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_inv_srl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_gl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[msk] <= 1
    GL[msk] <= RL()

    RL[~msk] <= 0
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] ^= GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_gl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x1522)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_gl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_inv_gl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[msk] <= 1
    GL[msk] <= RL()

    RL[~msk] <= 0
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] ^= INV_GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_inv_gl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_inv_gl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_ggl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    GGL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] ^= GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_ggl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABC2)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_ggl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_inv_ggl_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    GGL[::] <= RL()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] ^= INV_GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_inv_ggl_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x152D)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_inv_ggl_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_rsp16_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RSP16[::] <= RL()
    RSP_START_RET()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] ^= RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_rsp16_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x1522)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_rsp16_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_inv_rsp16_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RL[msk] <= 1
    RSP16[::] <= RL()
    RSP_START_RET()
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src()
    RL[msk] ^= INV_RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_inv_rsp16_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_inv_rsp16_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_sb_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] |= src()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_sb_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBFEF)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_sb_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp

# FIXME: This test is broken. It passed at one time but it seems that happened
# because the state of the APU was changed with other, unintentional RWINH
# settings that I have had difficulty reproducing.

# @belex_apl
# def fill_rl_and_eq_sb_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
#     RL[~msk] <= 0
#     dst[~msk] <= RL()

#     RL[msk] <= mrk()
#     RWINH_SET[msk]
#     RL[::] &= src()
#     dst[::] <= RL()
#     RWINH_RST[msk]


# @parameterized_belex_test
# def test_fill_rl_and_eq_sb_w_rwinh(diri: DIRI):
#     dst_vp = 0
#     src_vp = 1
#     mrk_vp = 2
#     msk_vp = 0xBEEF

#     expected_value = u16_to_bool(0xAACD)
#     expected_value[1::2, ::] = False

#     diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
#     diri.hb[mrk_vp, ::2, ::] = True

#     fill_rl_and_eq_sb_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

#     assert np.array_equal(expected_value, diri.hb[dst_vp])

#     return dst_vp

# FIXME: This test is broken. It passed at one time but it seems that happened
# because the state of the APU was changed with other, unintentional RWINH
# settings that I have had difficulty reproducing.

# @belex_apl
# def fill_rl_and_eq_inv_sb_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
#     RL[~msk] <= 0
#     dst[~msk] <= RL()

#     RL[msk] <= mrk()
#     RWINH_SET[msk]
#     RL[::] &= ~src()
#     dst[::] <= RL()
#     RWINH_RST[msk]


# @parameterized_belex_test
# def test_fill_rl_and_eq_inv_sb_w_rwinh(diri: DIRI):
#     dst_vp = 0
#     src_vp = 1
#     mrk_vp = 2
#     msk_vp = 0xBEEF

#     expected_value = u16_to_bool(0x1422)
#     expected_value[1::2, ::] = False

#     diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
#     diri.hb[mrk_vp, ::2, ::] = True

#     fill_rl_and_eq_inv_sb_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

#     assert np.array_equal(expected_value, diri.hb[dst_vp])

#     return dst_vp


@belex_apl
def fill_rl_xor_eq_sb_w_rwinh(Belex, dst: VR, src: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    dst[~msk] <= RL()

    RWINH_SET[RL[msk] <= mrk()]
    RL[::] ^= src()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_sb_w_rwinh(diri: DIRI):
    dst_vp = 0
    src_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x1522)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_sb_w_rwinh(dst_vp, src_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_sb_w_rsp16_w_rwinh(Belex, dst: VR, mrk: VR, msk: Mask):
    RL[::] <= 1
    RSP16[::] <= RL()
    RSP_START_RET()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    dst[::] <= RSP16()
    RSP_END()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_sb_w_rsp16_w_rwinh(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x4110)
    expected_value[0::2, ::] = True

    diri.hb[mrk_vp, ::2, ::] = True

    fill_sb_w_rsp16_w_rwinh(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_sb_w_inv_rsp16_w_rwinh(Belex, dst: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    dst[::] <= INV_RSP16()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_sb_w_inv_rsp16_w_rwinh(diri: DIRI):
    dst_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x4110)
    expected_value[0::2, ::] = True

    diri.hb[mrk_vp, ::2, ::] = True

    fill_sb_w_inv_rsp16_w_rwinh(dst_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_w_sb_and_gl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src() & GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_w_sb_and_gl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0000)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_w_sb_and_gl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_w_sb_and_inv_gl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src() & INV_GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_w_sb_and_inv_gl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xABCD)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_w_sb_and_inv_gl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_w_sb_and_ggl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GGL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src() & GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_w_sb_and_ggl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x00C0)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_w_sb_and_ggl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_w_sb_and_inv_ggl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GGL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src() & INV_GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_w_sb_and_inv_ggl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xAB0D)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_w_sb_and_inv_ggl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_w_sb_and_rsp16_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    RSP16[::] <= RL()
    RSP_START_RET()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src() & RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_w_sb_and_rsp16_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x29C4)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_w_sb_and_rsp16_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_w_sb_and_inv_rsp16_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    RSP16[::] <= RL()
    RSP_START_RET()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= src() & INV_RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_w_sb_and_inv_rsp16_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x8209)
    expected_value[1::2, ::] = u16_to_bool(0x0000, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_w_sb_and_inv_rsp16_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_sb_and_gl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] |= src() & GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_sb_and_gl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_sb_and_gl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_sb_and_inv_gl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] |= src() & INV_GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_sb_and_inv_gl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBFEF)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_sb_and_inv_gl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_sb_and_ggl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GGL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] |= src() & GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_sb_and_ggl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_sb_and_ggl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_sb_and_inv_ggl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GGL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] |= src() & INV_GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_sb_and_inv_ggl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBFEF)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_sb_and_inv_ggl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_sb_and_rsp16_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    RSP16[::] <= RL()
    RSP_START_RET()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] |= src() & RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_sb_and_rsp16_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBFEF)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_sb_and_rsp16_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_or_eq_sb_and_inv_rsp16_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    RSP16[::] <= RL()
    RSP_START_RET()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] |= src() & INV_RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_or_eq_sb_and_inv_rsp16_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_or_eq_sb_and_inv_rsp16_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_sb_and_gl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] &= src() & GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_sb_and_gl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x0000)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_sb_and_gl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_sb_and_inv_gl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] &= src() & INV_GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_sb_and_inv_gl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xAACD)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_sb_and_inv_gl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_sb_and_ggl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GGL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] &= src() & GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_sb_and_ggl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x00C0)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_sb_and_ggl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_and_eq_sb_and_inv_ggl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GGL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] &= src() & INV_GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_sb_and_inv_ggl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xAA0D)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_sb_and_inv_ggl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp

# FIXME: This test is broken. It passed at one time but it seems that happened
# because the state of the APU was changed with other, unintentional RWINH
# settings that I have had difficulty reproducing.

# @belex_apl
# def fill_rl_and_eq_sb_and_rsp16_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
#     RL[::] <= tmp()
#     RSP16[::] <= RL()
#     RSP_START_RET()
#     RL[~msk] <= 0
#     RWINH_SET[RL[msk] <= mrk()]
#     RL[::] &= src() & RSP16()
#     RSP_END()
#     dst[::] <= RL()
#     RWINH_RST[msk]


# @parameterized_belex_test
# def test_fill_rl_and_eq_sb_and_rsp16_w_rwinh(diri: DIRI) -> int:
#     dst_vp = 0
#     src_vp = 1
#     tmp_vp = 2
#     mrk_vp = 3
#     msk_vp = 0xBEEF

#     expected_value = u16_to_bool(0x28C4)
#     expected_value[1::2, ::] = False

#     diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
#     diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
#     diri.hb[mrk_vp, ::2, ::] = True

#     fill_rl_and_eq_sb_and_rsp16_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

#     assert np.array_equal(expected_value, diri.hb[dst_vp])

#     return dst_vp


@belex_apl
def fill_rl_and_eq_sb_and_inv_rsp16_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    RSP16[::] <= RL()
    RSP_START_RET()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] &= src() & INV_RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_and_eq_sb_and_inv_rsp16_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x8209)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_and_eq_sb_and_inv_rsp16_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_sb_and_gl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] ^= src() & GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_sb_and_gl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBEEF)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_sb_and_gl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_sb_and_inv_gl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] ^= src() & INV_GL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_sb_and_inv_gl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x1522)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_sb_and_inv_gl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_sb_and_ggl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GGL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] ^= src() & GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_sb_and_ggl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0xBE2F)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_sb_and_ggl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_sb_and_inv_ggl_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    GGL[::] <= RL()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] ^= src() & INV_GGL()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_sb_and_inv_ggl_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x15E2)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_sb_and_inv_ggl_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_sb_and_rsp16_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    RSP16[::] <= RL()
    RSP_START_RET()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] ^= src() & RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_sb_and_rsp16_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x972B)
    expected_value[1::2, ::] = u16_to_bool(0x0100, num_plats=1)

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_sb_and_rsp16_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def fill_rl_xor_eq_sb_and_inv_rsp16_w_rwinh(Belex, dst: VR, src: VR, tmp: VR, mrk: VR, msk: Mask):
    RL[::] <= tmp()
    RSP16[::] <= RL()
    RSP_START_RET()
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] ^= src() & INV_RSP16()
    RSP_END()
    dst[::] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_fill_rl_xor_eq_sb_and_inv_rsp16_w_rwinh(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    tmp_vp = 2
    mrk_vp = 3
    msk_vp = 0xBEEF

    expected_value = u16_to_bool(0x3CE6)
    expected_value[1::2, ::] = False

    diri.hb[src_vp, ::, ::] = u16_to_bool(0xABCD, num_plats=1)
    diri.hb[tmp_vp, ::, ::] = u16_to_bool(0x39F6, num_plats=1)
    diri.hb[mrk_vp, ::2, ::] = True

    fill_rl_xor_eq_sb_and_inv_rsp16_w_rwinh(dst_vp, src_vp, tmp_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[dst_vp])

    return dst_vp


@belex_apl
def write_b4_read_w_rwinh(Belex, out: VR, mrk: VR, msk: Mask):
    RWINH_SET[RL[msk] <= mrk()]
    out[msk] <= RL()
    RL[~msk] <= mrk()
    out[~msk] <= RL()
    RWINH_RST[msk]


@parameterized_belex_test
def test_write_b4_read_w_rwinh(diri: DIRI):
    out_vp = 0
    mrk_vp = 1
    msk_vp = 0xF00D

    diri.hb[mrk_vp] = u16_to_bool(0xBEEF, num_plats=1)
    diri.hb[mrk_vp, 1::2, ::] = False

    expected_value = deepcopy(diri.hb[mrk_vp])

    write_b4_read_w_rwinh(out_vp, mrk_vp, msk_vp)
    assert np.array_equal(diri.hb[out_vp], expected_value)

    return out_vp


# ===========================================================================
# TODO: Determine if this rule is different from the standard laning rule for
# combining READs and WRITEs.
# ===========================================================================

# @belex_apl
# def write_b4_read_in_same_instr_w_rwinh(Belex, out: VR, mrk: VR, msk: Mask):
#     RWINH_SET[RL[msk] <= mrk()]
#     # It is not allowed to combine WRITE and READ within the same instruction
#     # within the context of RWINH.
#     with apl_commands():
#         out[msk] <= RL()
#         RL[~msk] <= mrk()
#     out[~msk] <= RL()
#     RWINH_RST[msk]


# @parameterized_belex_test(generate_code=False)
# def test_write_b4_read_in_same_instr_w_rwinh(diri: DIRI) -> None:
#     out_vp = 0
#     mrk_vp = 1
#     msk_vp = 0xF00D

#     diri.hb[mrk_vp] = u16_to_bool(0xBEEF, num_plats=1)
#     diri.hb[mrk_vp, 1::2, ::] = False

#     fragment_caller_call = \
#         write_b4_read_in_same_instr_w_rwinh(out_vp, mrk_vp, msk_vp)
#     vm = BLEIRVirtualMachine(interpret=False, generate_code=False)
#     with pytest.raises(BLEIR.SemanticError):
#         vm.compile(fragment_caller_call)


@belex_apl
def write_gl_1s_over_vr_w_rwinh(Belex, out: VR, mrk: VR, msk: Mask):
    RL[::] <= 1
    GL[::] <= RL()
    RL[::] <= 0
    out[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    RWINH_RST[msk]
    out[::] <= GL()


@parameterized_belex_test
def test_write_gl_1s_over_vr_w_rwinh(diri: DIRI) -> int:
    out_vp = 0
    mrk_vp = 1
    msk_vp = 0xF00D

    diri.hb[mrk_vp] = u16_to_bool(0xBEEF, num_plats=1)
    diri.hb[mrk_vp, 1::2, ::] = False

    write_gl_1s_over_vr_w_rwinh(out_vp, mrk_vp, msk_vp)
    # assert diri.hb[out_vp].all()

    return out_vp


@belex_apl
def write_inv_rsp16_over_vr_w_rwinh(Belex, out: VR, mrk: VR, msk: Mask):
    RL[::] <= 0
    out[::] <= RL()
    RWINH_SET[RL[msk] <= mrk()]
    out[::] <= INV_RSP16()
    RWINH_RST[msk]


@parameterized_belex_test
def test_write_inv_rsp16_over_vr_w_rwinh(diri: DIRI) -> int:
    out_vp = 0
    mrk_vp = 1
    msk_vp = 0xF00D

    expected_value = diri.build_vr()
    expected_value[0::2] = u16_to_bool(0xBFFF, num_plats=1)
    expected_value[1::2] = u16_to_bool(0x0FF2, num_plats=1)

    diri.hb[mrk_vp] = u16_to_bool(0xBEEF, num_plats=1)
    diri.hb[mrk_vp, 1::2, ::] = False

    write_inv_rsp16_over_vr_w_rwinh(out_vp, mrk_vp, msk_vp)
    assert np.array_equal(diri.hb[out_vp], expected_value)

    return out_vp


@belex_apl
def write_part_unmasked_rl_to_vr_w_rwinh(Belex, dst: VR, mrk: VR):
    RL[::] <= 0
    dst[::] <= RL()
    RWINH_SET[RL["0xBEEF"] <= mrk()]
    RWINH_RST["0xABCD"]
    RL[::] <= 1
    dst[::] <= RL()
    RWINH_RST["0x1422"]


@parameterized_belex_test
def test_write_part_unmasked_rl_to_vr_w_rwinh(diri: DIRI):
    out_vp = 0
    mrk_vp = 1

    expected_value = diri.build_vr()
    expected_value[0::2] = u16_to_bool(0xFFFF, num_plats=1)
    expected_value[1::2] = u16_to_bool(0xEBDD, num_plats=1)

    diri.hb[mrk_vp] = u16_to_bool(0xBEEF, num_plats=1)
    diri.hb[mrk_vp, 1::2, ::] = False

    write_part_unmasked_rl_to_vr_w_rwinh(out_vp, mrk_vp)
    assert np.array_equal(diri.hb[out_vp], expected_value)

    return out_vp


@belex_apl
def read_w_rwinh_rst(Belex, out: VR, data: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RWINH_RST[RL[::] <= data()]
    out[::] <= RL()


@parameterized_belex_test
def test_read_w_rwinh_rst(diri: DIRI) -> int:
    out_vp = 0
    data_vp = 1
    mrk_vp = 2
    msk_vp = 0xBEEF

    data = vr_strategy().example()
    diri.hb[data_vp] = convert_to_bool(data)

    diri.hb[mrk_vp] = u16_to_bool(0xF00D)
    diri.hb[mrk_vp, 1::2, ::] = False

    expected_value = diri.hb[data_vp] \
        & (diri.hb[mrk_vp]
           & u16_to_bool(msk_vp, num_plats=1)
           | u16_to_bool(0xFFFF - msk_vp, num_plats=1))

    read_w_rwinh_rst(out_vp, data_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[out_vp])

    return out_vp


@belex_apl
def incremental_rwinh_rst(Belex, out: VR, mrk: VR):
    out[::] <= RSP16()
    RWINH_SET[RL["0xBEEF"] <= mrk()]
    RWINH_RST["0xB000"]
    RWINH_RST["0x0E00"]
    RWINH_RST["0x00E0"]
    RWINH_RST["0x000F"]
    out[::] <= INV_RSP16()


@parameterized_belex_test
def test_incremental_rwinh_rst(diri: DIRI) -> int:
    out_vp = 0
    mrk_vp = 1

    diri.hb[mrk_vp] = u16_to_bool(0xF00D)
    diri.hb[mrk_vp, 1::2, ::] = False

    incremental_rwinh_rst(out_vp, mrk_vp)

    assert diri.hb[out_vp].all()

    return out_vp


@belex_apl
def broadcast_w_rwinh_rst(Belex, out: VR, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    temp = Belex.VR()
    temp[::] <= INV_RL()
    with apl_commands():
        RSP16[::] <= RL()
        RWINH_RST[RL[::] <= temp()]
    RSP_START_RET()
    out[::] <= RSP16()
    RSP_END()


@parameterized_belex_test
def test_broadcast_w_rwinh_rst(diri: DIRI) -> int:
    out_vp = 0
    mrk_vp = 1
    msk_vp = 0xBEEF

    diri.hb[mrk_vp] = u16_to_bool(0xF00D)
    diri.hb[mrk_vp, 1::2, ::] = False

    expected_value = u16_to_bool(0xB00D)

    broadcast_w_rwinh_rst(out_vp, mrk_vp, msk_vp)

    assert np.array_equal(expected_value, diri.hb[out_vp])

    return out_vp


@belex_apl
def write_from_gl_w_read_in_rwinh_set(Belex, out: VR, mrk: VR):
    RL[::] <= mrk()
    GL[::] <= RL()
    tmp = Belex.VR()
    tmp["0xFF00"] <= GL()
    with apl_commands():
        tmp["0x00FF"] <= GL()
        RL["0xFF00"] <= tmp()
        RWINH_SET["0xFF00"]
    with apl_commands():
        RL["0x00FF"] <= tmp()
        RWINH_SET["0x00FF"]
    out[::] <= INV_RSP16()
    RWINH_RST["0xFFFF"]


@parameterized_belex_test
def test_write_from_gl_w_read_in_rwinh_set(diri: DIRI):
    out_vp = 0
    mrk_vp = 1

    diri.hb[mrk_vp] = diri.build_vr()
    diri.hb[mrk_vp, 0::2, ::] = True

    write_from_gl_w_read_in_rwinh_set(out_vp, mrk_vp)

    assert diri.hb[out_vp, 0::2, ::].all()
    assert not diri.hb[out_vp, 1::2, ::].any()

    return out_vp


@belex_apl
def ggl_from_rl(Belex, out: VR, msk: Mask):
    GGL[msk] <= RL()
    out[::] <= GGL()


@parameterized_belex_test
def test_ggl_from_rl_w_empty_mask(diri: DIRI) -> int:
    out_vp = 0
    msk_vp = 0x0000
    ggl_from_rl(out_vp, msk_vp)
    assert diri.hb[out_vp].all()
    return out_vp


@belex_apl
def gl_from_rl(Belex, out: VR, msk: Mask):
    GL[msk] <= RL()
    out[::] <= GL()


@parameterized_belex_test
def test_gl_from_rl_w_empty_mask(diri: DIRI) -> int:
    out_vp = 0
    msk_vp = 0x0000
    gl_from_rl(out_vp, msk_vp)
    assert diri.hb[out_vp].all()
    return out_vp


@belex_apl
def rl_from_empty_sb(Belex, out: VR, vrs: RE):
    RL[::] <= vrs()
    out[::] <= RL()


@parameterized_belex_test
def test_rl_from_empty_sb(diri: DIRI) -> int:
    out_vp = 0
    vrs_vp = 0x000000
    rl_from_empty_sb(out_vp, vrs_vp)
    assert diri.hb[out_vp].all()
    return out_vp


@belex_apl
def rl_from_nrl_and_srl(Belex, out: VR, data: VR):
    RL[::] <= data()
    # The following commands must be executed in parallel for correct results:
    with apl_commands():
        RL[SM_0X1111] <= SRL()
        RL[SM_0X1111 << 1] <= NRL()
    out[::] <= RL()


@parameterized_belex_test
def test_rl_from_nrl_and_srl(diri: DIRI) -> int:
    out_vp = 0
    data_vp = 1

    diri.repeatably_randomize_half_bank()

    iout = convert_to_u16(diri.hb[data_vp])
    nout = iout << 1
    sout = iout >> 1
    eout = (iout & (0xFFFF - 0x3333)) | (nout & 0x2222) | (sout & 0x1111)
    expected_value = convert_to_bool(eout)

    rl_from_nrl_and_srl(out_vp, data_vp)

    assert np.array_equal(diri.hb[out_vp], expected_value)

    return out_vp


@belex_apl
def src_to_rsp32k_to_dst(Belex, dst: VR, src: VR):
    RL[::] <= src()
    RSP16[::] <= RL()
    RSP256() <= RSP16()
    RSP2K() <= RSP256()
    RSP32K() <= RSP2K()
    RSP_START_RET()
    RSP2K() <= RSP32K()
    RSP256() <= RSP2K()
    RSP16() <= RSP256()
    dst[::] <= RSP16()
    RSP_END()


@parameterized_belex_test
def test_src_to_rsp32k_to_dst(diri: DIRI) -> int:
    dst_vp = 0
    src_vp = 1
    diri.hb[src_vp, range(8 * 2048, 16 * 2048)] = False
    diri.hb[src_vp, range(8 * 2048)] = \
        u16_to_bool(0xBEEF, num_plats=(8 * 2048))
    src_to_rsp32k_to_dst(dst_vp, src_vp)
    assert diri.hb[dst_vp, range(8 * 2048)].all()
    assert not diri.hb[dst_vp, range(8 * 2048, 16 * 2048)].any()
    return dst_vp


@belex_apl
def broadcast_to_ggl_and_gl_from_overwritten_secs(Belex, dst: VR,
                                                  expected_ggl: VR,
                                                  expected_gl: VR):
    with apl_commands():
        RL[SM_0XFFFF] <= RSP16()
    dst[SM_0XFFFF] <= RL()
    with apl_commands():
        RL[SM_0X0001 << 13] <= INV_RSP16()
        RL[SM_0X0001 << 14] <= INV_RSP16()
    with apl_commands():
        GGL[SM_0X0001 << 12] <= RL()
        GGL[SM_0X0001 << 13] <= RL()
    with apl_commands():
        GL[SM_0X0001 << 12] <= RL()
        GL[SM_0X0001 << 13] <= RL()
    Belex.glass(RL, plats = range(0, 16, 1), order = "lsb")
    Belex.glass(GGL, plats = range(0, 16, 1), order = "lsb")
    Belex.glass(GL, plats = range(0, 16, 1), order = "lsb")
    with apl_commands():
        NOOP()
    RL[SM_0XFFFF] <= expected_ggl() ^ GGL()
    dst[SM_0XFFFF] <= RL()
    RL[SM_0XFFFF] <= expected_gl() ^ GL()
    dst[SM_0XFFFF] |= RL()


@parameterized_belex_test
def test_broadcast_to_ggl_and_gl_from_overwritten_secs(diri: DIRI) -> int:
    dst_vp = 0
    expected_ggl_vp = 1
    expected_gl_vp = 2
    diri.hb[expected_ggl_vp] = u16_to_bool(0x0FFF, num_plats=1)
    diri.hb[expected_gl_vp] = u16_to_bool(0x0000, num_plats=1)
    broadcast_to_ggl_and_gl_from_overwritten_secs(dst_vp,
                                                  expected_ggl_vp,
                                                  expected_gl_vp)
    assert not diri.hb[dst_vp].any()
    return dst_vp


@belex_apl
def combine_gl_and_ggl_stmts(Belex, dst: VR,
                             expected_ggl: VR,
                             expected_gl: VR):
    with apl_commands():
        RL[SM_0XFFFF] <= RSP16()
    with apl_commands():
        RL[SM_0X0001 << 10] <= INV_RSP16()
        RL[SM_0X0001 << 11] <= INV_RSP16()
    with apl_commands():
        RN_REG_T0[SM_0XFFFF] <= RL()
    with apl_commands():
        RL[SM_0XFFFF << 9] <= RN_REG_T0()
        GGL[SM_0X0001 << 9] <= RL()
        GGL[SM_0X0001 << 10] <= RL()
        GL[SM_0X000F << 9] <= RL()
    RL[::] <= expected_ggl() ^ GGL()
    dst[::] <= RL()
    RL[::] <= expected_gl() ^ GL()
    dst[::] |= RL()


@parameterized_belex_test
def test_combine_gl_and_ggl_stmts(diri: DIRI) -> int:
    dst_vp = 0
    expected_ggl_vp = 1
    expected_gl_vp = 2
    diri.hb[expected_ggl_vp] = u16_to_bool(0xF0FF, num_plats=1)
    diri.hb[expected_gl_vp] = u16_to_bool(0x0000, num_plats=1)
    combine_gl_and_ggl_stmts(dst_vp, expected_ggl_vp, expected_gl_vp)
    assert not diri.hb[dst_vp].any()
    return dst_vp


@belex_apl
def neg_inv_gl_and_inv_gl_same_instr(Belex, in_out: VR):
    with apl_commands():
        RL[SM_0XFFFF] <= in_out()
        GL[SM_0XFFFF] <= RL()
    with apl_commands():
        RN_REG_T0[~(SM_0XFFFF << 14)] <= INV_GL()
        RL[~(SM_0XFFFF << 14)] <= in_out() & ~INV_GL()
    in_out[~(SM_0XFFFF << 14)] <= RL()


@parameterized_belex_test()
def test_neg_inv_gl_and_inv_gl_same_instr(diri: DIRI) -> int:
    in_out_vp = 0
    diri.hb[in_out_vp, 0::2] = True
    diri.hb[in_out_vp, 1::2] = False
    neg_inv_gl_and_inv_gl_same_instr(in_out_vp)
    assert diri.hb[in_out_vp, 0::2].all()
    assert not diri.hb[in_out_vp, 1::2].any()
    return in_out_vp
