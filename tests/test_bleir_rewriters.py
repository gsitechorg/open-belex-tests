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

import numpy as np

import open_belex.literal as LLB
from open_belex.bleir.allocators import AllocateRegisters
from open_belex.bleir.analyzers import (MAX_FRAGMENT_INSTRUCTIONS,
                                        NumFragmentInstructionsAnalyzer,
                                        RegisterScanner)
from open_belex.bleir.interpreters import BLEIRInterpreter
from open_belex.bleir.rewriters import (AutomaticLaner,
                                        EnsureNoEmptyFragBodies,
                                        InjectMissingNOOPs,
                                        PartitionFragmentsIntoDigestibleChunks,
                                        RewriteSingletonMultiStatement)
from open_belex.bleir.types import (INV_GL, NOOP, RN_REG, SB, SM_REG,
                                    STATEMENT, AllocatedRegister, Fragment,
                                    FragmentCaller, GlassStatement,
                                    LineComment, MultiStatement, assign,
                                    masked, multi_statement, statement)
from open_belex.bleir.walkables import BLEIRWalker
from open_belex.diri.half_bank import DIRI
from open_belex.literal import (GGL, GL, INV_RSP16, RL, RSP2K, RSP16, RSP256,
                                RSP_END, RSP_START_RET, RWINH_RST, RWINH_SET,
                                VR, apl_commands, belex_apl, u16)
from open_belex.utils.example_utils import u16_to_bool

from open_belex_libs.common import reset_16

from open_belex_tests.utils import parameterized_belex_test


def test_rewrite_singleton_multi_statement():
    walker = BLEIRWalker()
    rewrite_singleton_multi_statement = RewriteSingletonMultiStatement()

    fragment = Fragment(
        identifier="fragment",
        parameters=[],
        operations=[
            statement(NOOP),
            multi_statement([
                statement(NOOP),
                statement(NOOP),
            ]),
            statement(NOOP),
            statement(NOOP),
        ])

    assert walker.walk(rewrite_singleton_multi_statement, fragment) == Fragment(
        identifier="fragment",
        parameters=[],
        operations=[
            multi_statement([
                statement(NOOP),
            ]),
            multi_statement([
                statement(NOOP),
                statement(NOOP),
            ]),
            multi_statement([
                statement(NOOP),
            ]),
            multi_statement([
                statement(NOOP),
            ]),
        ])


def test_allocate_registers():
    walker = BLEIRWalker()
    register_scanner = RegisterScanner()
    shared_registers_by_frag = {}
    allocate_registers = AllocateRegisters(shared_registers_by_frag)

    lvr_rp = RN_REG("lvr")
    msk_rp = SM_REG("msk")

    fragment = Fragment(
        identifier="fragment",
        parameters=[lvr_rp, msk_rp],
        operations=[
            statement(masked(msk_rp, assign(SB[lvr_rp], RL))),
        ])

    fragment_caller = FragmentCaller(fragment=fragment)
    walker.walk(register_scanner, fragment_caller)
    assert walker.walk(allocate_registers, fragment_caller) == FragmentCaller(
        fragment=fragment,
        registers=[
            AllocatedRegister(
                parameter=lvr_rp,
                register="RN_REG_0"),
            AllocatedRegister(
                parameter=msk_rp,
                register="SM_REG_0"),
        ])


def test_ensure_no_empty_frag_bodies():
    walker = BLEIRWalker()
    analyzer = NumFragmentInstructionsAnalyzer()
    ensure_no_empty_frag_bodies = EnsureNoEmptyFragBodies(
        num_fragment_instructions_analyzer=analyzer)

    fragment = Fragment(
        identifier="fragment",
        parameters=[],
        operations=[])

    walker.walk(analyzer, fragment)
    assert walker.walk(ensure_no_empty_frag_bodies, fragment) == Fragment(
        identifier="fragment",
        parameters=tuple(),
        operations=(
            statement(NOOP),
        ))


def test_inject_missing_noops():
    walker = BLEIRWalker()
    inject_missing_noops = InjectMissingNOOPs()

    msk_rp = SM_REG("msk")

    fragment = Fragment(
        identifier="fragment",
        parameters=[msk_rp],
        operations=[
            statement(masked(msk_rp, assign(GL, RL))),
            statement(masked(msk_rp, assign(RL, GL))),
            statement(NOOP),
            statement(masked(msk_rp, assign(RL, INV_GL))),
        ])

    assert walker.walk(inject_missing_noops, fragment) == Fragment(
        identifier="fragment",
        parameters=[msk_rp],
        operations=[
            statement(masked(msk_rp, assign(GL, RL))),
            statement(NOOP),
            statement(masked(msk_rp, assign(RL, GL))),
            statement(NOOP),
            statement(masked(msk_rp, assign(RL, INV_GL))),
            statement(NOOP),
        ])


### =============================================================================
### Regression test for avoiding a fragment with a single statement, that happens
### to be only glass and generates an empty frag body.
### =============================================================================


@belex_apl
def two_noop_partitions_and_singleton_glass_partition(Belex):
    for noop_num in range(2 * MAX_FRAGMENT_INSTRUCTIONS):
        LLB.NOOP()
    Belex.comment("=== BEGIN ===")
    Belex.glass(LLB.RL, plats=1, sections=1)
    Belex.comment("=== END ===")


@parameterized_belex_test(interpret=False)
def test_partitioner_against_singleton_glass_partition(diri: DIRI) -> int:
    out = 0
    reset_16(out)

    walker = BLEIRWalker()
    analyzer = NumFragmentInstructionsAnalyzer()
    partitioner = PartitionFragmentsIntoDigestibleChunks(
        num_fragment_instructions_analyzer=analyzer)

    call = two_noop_partitions_and_singleton_glass_partition()
    call = walker.walk(analyzer, call)
    call = walker.walk(partitioner, call)

    assert call.fragment.children is not None
    assert len(call.fragment.children) == 2
    assert len(call.fragment.children[0].operations) == MAX_FRAGMENT_INSTRUCTIONS
    assert len(call.fragment.children[1].operations) == MAX_FRAGMENT_INSTRUCTIONS + 3
    assert isinstance(call.fragment.children[1].operations[-1], LineComment.__args__)
    assert isinstance(call.fragment.children[1].operations[-2], STATEMENT)
    assert isinstance(call.fragment.children[1].operations[-2].operation, GlassStatement)
    assert isinstance(call.fragment.children[1].operations[-3], LineComment.__args__)

    return out


def test_automatic_laner():
    automatic_laner = AutomaticLaner()
    walker = BLEIRWalker()

    diri = DIRI()
    interpreter = BLEIRInterpreter(diri=diri)

    # ======================
    # Test Write-before-Read
    # ======================

    @belex_apl
    def write_b4_read_no_lane(Belex, src: VR):
        src["0x00FF"] <= RSP16()
        RL["0xFF00"] <= src()

    @belex_apl
    def write_b4_read_w_lane(Belex, src: VR):
        with apl_commands():
            # Cannot write-to and read-from the same sections of the same VR
            # within the same instruction.
            src["0x00FF"] <= RSP16()
            RL["0xFF00"] <= src()

    src_vp = 1
    expected_value = walker.walk(automatic_laner,
                                 write_b4_read_w_lane(src_vp))
    actual_value = walker.walk(automatic_laner,
                               write_b4_read_no_lane(src_vp))
    assert expected_value.fragment.operations == actual_value.fragment.operations

    # ==========================
    # Test Read-before-Broadcast
    # ==========================

    @belex_apl
    def read_b4_broadcast_no_lane(Belex):
        RL[::] <= RSP16()
        GGL[::] <= RL()

    @belex_apl
    def read_b4_broadcast_w_lane(Belex):
        with apl_commands():
            RL[::] <= RSP16()
            GGL[::] <= RL()

    expected_value = walker.walk(automatic_laner,
                                 read_b4_broadcast_w_lane())
    actual_value = walker.walk(automatic_laner,
                               read_b4_broadcast_no_lane())
    assert expected_value.fragment.operations == actual_value.fragment.operations

    # ==================================
    # Test disjoint, parameterized Reads
    # ==================================

    @belex_apl
    def disjoint_param_reads(Belex, out: VR, val: u16):
        RL[val] <= 1
        RL[~val] <= 0
        out[::] <= RL()

        # ^^^ This will optimize to the following:
        # ----------------------------------------
        # with apl_commands():
        #     RL[val] <= 1
        #     RL[~val] <= 0
        # out[::] <= RL()

    out_vp = 0
    val = 0xABCD
    expected_value = u16_to_bool(val)

    # Unoptimized validations:
    # ------------------------

    fragment_caller_call = disjoint_param_reads(out_vp, val)
    fragment = fragment_caller_call.fragment
    assert len(fragment.operations) == 3

    diri.repeatably_randomize_half_bank()
    assert not np.array_equal(diri.hb[out_vp], expected_value)

    interpreter.visit_fragment_caller_call(fragment_caller_call)
    assert np.array_equal(diri.hb[out_vp], expected_value)

    # Optimized validations:
    # ----------------------

    fragment_caller_call = walker.walk(automatic_laner, fragment_caller_call)
    fragment = fragment_caller_call.fragment
    assert len(fragment.operations) == 2

    diri.repeatably_randomize_half_bank()
    assert not np.array_equal(diri.hb[out_vp], expected_value)

    interpreter.visit_fragment_caller_call(fragment_caller_call)
    assert np.array_equal(diri.hb[out_vp], expected_value)

    # =========================================
    # Test writes to two different destinations
    # =========================================

    @belex_apl
    def write_write_diff_dsts(Belex, dst1: VR, dst2: VR, val: u16):
        with apl_commands():
            RL[val] <= 1
            RL[~val] <= 0
        dst1[::] <= RL()
        dst2[::] <= RL()

    dst1_vp = 0
    dst2_vp = 1
    val = 0xBEEF

    expected_value = u16_to_bool(val)

    # Unoptimized validations:
    # ------------------------

    fragment_caller_call = write_write_diff_dsts(dst1_vp, dst2_vp, val)
    fragment = fragment_caller_call.fragment
    assert len(fragment.operations) == 3

    diri.repeatably_randomize_half_bank()
    assert not np.array_equal(diri.hb[dst1_vp], expected_value)
    assert not np.array_equal(diri.hb[dst2_vp], expected_value)

    interpreter.visit_fragment_caller_call(fragment_caller_call)
    assert np.array_equal(diri.hb[dst1_vp], expected_value)
    assert np.array_equal(diri.hb[dst2_vp], expected_value)

    # Optimized validations:
    # ----------------------

    fragment_caller_call = walker.walk(automatic_laner, fragment_caller_call)
    fragment = fragment_caller_call.fragment
    assert len(fragment.operations) == 2

    diri.repeatably_randomize_half_bank()
    assert not np.array_equal(diri.hb[dst1_vp], expected_value)
    assert not np.array_equal(diri.hb[dst2_vp], expected_value)

    interpreter.visit_fragment_caller_call(fragment_caller_call)
    assert np.array_equal(diri.hb[dst1_vp], expected_value)
    assert np.array_equal(diri.hb[dst2_vp], expected_value)

    # ==================================
    # Test writes with disjoint sections
    # ==================================

    @belex_apl
    def disjoint_writes(Belex, dst: VR):
        dst["0xFFFF"] <= RSP16()
        dst["0x00FF"] <= INV_RSP16()
        dst["0xFF00"] <= INV_RSP16()

    dst_vp = 0
    expected_value = u16_to_bool(0xFFFF)

    # Unoptimized validations:
    # ------------------------

    fragment_caller_call = disjoint_writes(dst_vp)
    fragment = fragment_caller_call.fragment
    assert len(fragment.operations) == 3

    diri.repeatably_randomize_half_bank()
    assert not np.array_equal(diri.hb[dst_vp], expected_value)

    interpreter.visit_fragment_caller_call(fragment_caller_call)
    assert np.array_equal(diri.hb[dst_vp], expected_value)

    # Optimized validations:
    # ----------------------

    fragment_caller_call = walker.walk(automatic_laner, fragment_caller_call)
    fragment = fragment_caller_call.fragment
    assert len(fragment.operations) == 2

    diri.repeatably_randomize_half_bank()
    assert not np.array_equal(diri.hb[dst_vp], expected_value)

    interpreter.visit_fragment_caller_call(fragment_caller_call)
    assert np.array_equal(diri.hb[dst_vp], expected_value)

    # ===========================================
    # Test writes that share at least one section
    # ===========================================

    @belex_apl
    def overlapping_writes(Belex, dst: VR):
        dst["0xFFFF"] <= RSP16()
        dst["0xBEEF"] <= INV_RSP16()
        dst["0xF00D"] <= INV_RSP16()

    dst_vp = 0
    expected_value = u16_to_bool(0xBEEF | 0xF00D)

    # Unoptimized validations:
    # ------------------------

    fragment_caller_call = overlapping_writes(dst_vp)
    fragment = fragment_caller_call.fragment
    assert len(fragment.operations) == 3

    diri.repeatably_randomize_half_bank()
    assert not np.array_equal(diri.hb[dst_vp], expected_value)

    interpreter.visit_fragment_caller_call(fragment_caller_call)
    assert np.array_equal(diri.hb[dst_vp], expected_value)

    # Optimized validations:
    # ----------------------

    fragment_caller_call = walker.walk(automatic_laner, fragment_caller_call)
    fragment = fragment_caller_call.fragment
    assert len(fragment.operations) == 3

    diri.repeatably_randomize_half_bank()
    assert not np.array_equal(diri.hb[dst_vp], expected_value)

    interpreter.visit_fragment_caller_call(fragment_caller_call)
    assert np.array_equal(diri.hb[dst_vp], expected_value)

    # =================
    # Test RSP pipeline
    # =================

    @belex_apl
    def rsp_out_in_w_val(Belex, out: VR, val: u16):
        RL[val] <= 1
        RL[~val] <= 0
        RSP16[::] <= RL()
        RSP256() <= RSP16()
        RSP2K() <= RSP256()
        RSP_START_RET()
        RSP256() <= RSP2K()
        RSP16() <= RSP256()
        out[::] <= RSP16()
        RSP_END()

    out_vp = 0
    val = 0x1234
    expected_value = u16_to_bool(val)

    # Unoptimized validations:
    # ------------------------

    fragment_caller_call = rsp_out_in_w_val(out_vp, val)
    fragment = fragment_caller_call.fragment
    assert len(fragment.operations) == 10

    diri.repeatably_randomize_half_bank()
    assert not np.array_equal(diri.hb[out_vp], expected_value)

    interpreter.visit_fragment_caller_call(fragment_caller_call)
    assert np.array_equal(diri.hb[out_vp], expected_value)

    # Optimized validations:
    # ----------------------

    fragment_caller_call = walker.walk(automatic_laner, fragment_caller_call)
    fragment = fragment_caller_call.fragment
    assert len(fragment.operations) == 9

    diri.repeatably_randomize_half_bank()
    assert not np.array_equal(diri.hb[out_vp], expected_value)

    interpreter.visit_fragment_caller_call(fragment_caller_call)
    assert np.array_equal(diri.hb[out_vp], expected_value)

    @belex_apl
    def write_read_instr(Belex, mrk: VR):
        tmp = Belex.VR()
        tmp[::] <= GL()
        RWINH_SET[RL[::] <= mrk()]
        tmp[::] <= GL()
        RL[::] <= mrk()
        RWINH_RST[::]

    mrk_vp = 0

    fragment_caller_call = write_read_instr(mrk_vp)
    fragment_caller_call = walker.walk(automatic_laner, fragment_caller_call)
    fragment = fragment_caller_call.fragment
    operations = fragment.operations

    assert len(operations) == 4

    assert isinstance(operations[0], STATEMENT)
    assert str(operations[0]) == "_INTERNAL_SM_0XFFFF: SB[_INTERNAL_VR_000] = GL;"

    assert isinstance(operations[1], STATEMENT)
    assert str(operations[1]) == "_INTERNAL_SM_0XFFFF: RL = SB[mrk] RWINH_SET;"

    assert isinstance(operations[2], MultiStatement)
    assert str(operations[2]) == "{" + " ".join([
        "_INTERNAL_SM_0XFFFF: SB[_INTERNAL_VR_000] = GL;",
        "_INTERNAL_SM_0XFFFF: RL = SB[mrk];",
    ]) + "}"

    assert isinstance(operations[3], STATEMENT)
    assert str(operations[3]) == "_INTERNAL_SM_0XFFFF: RWINH_RST;"
