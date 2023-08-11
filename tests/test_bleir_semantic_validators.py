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

import pytest

import open_belex.literal as LLB
from open_belex.bleir.semantic_validators import (FragmentCallerCallValidator,
                                                  MultiStatementValidator,
                                                  ParameterIDValidator)
from open_belex.bleir.types import (RL, RN_REG, RSP16, SB, SM_REG, Fragment,
                                    FragmentCaller, FragmentCallerCall,
                                    MultiStatement, SemanticError, assign,
                                    masked, statement)
from open_belex.bleir.virtual_machines import BLEIRVirtualMachine
from open_belex.bleir.walkables import BLEIRWalker
from open_belex.literal import VR, apl_commands, belex_apl


def test_parameter_id_validator():
    walker = BLEIRWalker()
    validator = ParameterIDValidator()

    ## ============== ##
    ## Valid Fragment ##
    ## ============== ##

    lvr_rp = RN_REG("lvr")
    rvr_rp = RN_REG("rvr")

    fs_rp = SM_REG("fs")
    sm_rp = SM_REG("sm")

    fragment = Fragment(
        identifier="fragment-1",
        parameters=[lvr_rp, rvr_rp, sm_rp],
        operations=[statement(masked(sm_rp, assign(RL, SB[lvr_rp, rvr_rp])))])

    # No error should be raised
    walker.walk(validator, fragment)

    ## ====================== ##
    ## Undeclared register id ##
    ## ====================== ##

    fragment = Fragment(
        identifier="fragment-1",
        parameters=[lvr_rp, rvr_rp, sm_rp],
        operations=[statement(masked(fs_rp, assign(RL, SB[lvr_rp, rvr_rp])))])

    with pytest.raises(SemanticError):
        walker.walk(validator, fragment)

    ## ======================= ##
    ## Conflicting register id ##
    ## ======================= ##

    lvr_sm_rp = SM_REG("lvr")

    fragment = Fragment(
        identifier="fragment-1",
        parameters=[lvr_rp, rvr_rp, lvr_sm_rp],
        operations=[statement(masked(lvr_sm_rp, assign(RL, SB[lvr_rp, rvr_rp])))])

    with pytest.raises(SemanticError):
        walker.walk(validator, fragment)


def test_fragment_caller_call_validator():
    walker = BLEIRWalker()
    validator = FragmentCallerCallValidator()

    ## ======================== ##
    ## Valid FragmentCallerCall ##
    ## ======================== ##

    lvr_rp = RN_REG("lvr")
    rvr_rp = RN_REG("rvr")

    sm_rp = SM_REG("sm")

    fragment = Fragment(
        identifier="fragment-1",
        parameters=[lvr_rp, rvr_rp, sm_rp],
        operations=[statement(masked(sm_rp, assign(RL, SB[lvr_rp, rvr_rp])))])

    fragment_caller = FragmentCaller(fragment=fragment)

    fragment_caller_call = FragmentCallerCall(
        caller=fragment_caller,
        parameters=[1, 2, 0xFFFE])

    walker.walk(validator, fragment_caller_call)

    ## ============================ ##
    ## Invalid number of parameters ##
    ## ============================ ##

    fragment_caller_call = FragmentCallerCall(
        caller=fragment_caller,
        parameters=[1, 2])

    with pytest.raises(SemanticError):
        walker.walk(validator, fragment_caller_call)

    fragment_caller_call = FragmentCallerCall(
        caller=fragment_caller,
        parameters=[1, 2, 0xFFFE, 3])

    with pytest.raises(SemanticError):
        walker.walk(validator, fragment_caller_call)

    ## ======================= ##
    ## Invalid parameter value ##
    ## ======================= ##

    fragment_caller_call = FragmentCallerCall(
        caller=fragment_caller,
        parameters=[1, 2, -0xFFFE])

    with pytest.raises(SemanticError):
        walker.walk(validator, fragment_caller_call)


def test_multi_statement_validator():
    walker = BLEIRWalker()
    validator = MultiStatementValidator()

    lvr_rp = RN_REG("lvr")
    rvr_rp = RN_REG("rvr")
    r2vr_rp = RN_REG("r2vr")

    fs_rp = SM_REG("fs")
    sm_rp = SM_REG("sm")
    sm2_rp = SM_REG("sm2")
    sm3_rp = SM_REG("sm3")

    statement_1 = statement(masked(sm_rp, assign(RL, SB[rvr_rp])))
    statement_2 = statement(masked(sm_rp, assign(SB[lvr_rp], RL)))
    statement_3 = statement(masked(fs_rp, assign(RL, SB[r2vr_rp])))
    statement_4 = statement(masked(fs_rp, assign(SB[lvr_rp], RL)))
    statement_5 = statement(masked(fs_rp, assign(RSP16, RL)))
    statement_6 = statement(masked(sm2_rp, assign(RL, SB[r2vr_rp])))
    statement_7 = statement(masked(sm3_rp, assign(RL, SB[r2vr_rp])))

    ## ========== ##
    ## Valid Case ##
    ## ========== ##

    multi_statement_0 = MultiStatement(statements=[])
    walker.walk(validator, multi_statement_0)

    ## ========== ##
    ## Valid Case ##
    ## ========== ##

    multi_statement_1 = MultiStatement(statements=[
        statement_1,
    ])

    walker.walk(validator, multi_statement_1)

    ## ========== ##
    ## Valid Case ##
    ## ========== ##

    multi_statement_2 = MultiStatement(statements=[
        statement_1,
        statement_2,
    ])

    walker.walk(validator, multi_statement_2)

    ## ========== ##
    ## Valid Case ##
    ## ========== ##

    multi_statement_3 = MultiStatement(statements=[
        statement_1,
        statement_2,
        statement_3,
    ])

    walker.walk(validator, multi_statement_3)

    ## ========== ##
    ## Valid Case ##
    ## ========== ##

    multi_statement_4 = MultiStatement(statements=[
        statement_1,
        statement_2,
        statement_3,
        statement_4,
    ])

    walker.walk(validator, multi_statement_4)

    ## ==== ##
    ## R2W1 ##
    ## ==== ##

    multi_statement_5 = MultiStatement(statements=[
        statement_1,
        statement_2,
        statement_3,
        statement_4,
        statement_5,
    ])

    walker.walk(validator, multi_statement_5)

    ## =================== ##
    ## Too Many Statements ##
    ## =================== ##

    multi_statement_6 = MultiStatement(statements=[
        statement_1,
        statement_2,
        statement_3,
        statement_4,
        statement_5,
        statement_6,
        statement_7,
    ])

    with pytest.raises(SemanticError):
        walker.walk(validator, multi_statement_6)


@belex_apl
def noop(Belex, x: VR):
    pass


def test_ensure_write_before_read():
    vm = BLEIRVirtualMachine(generate_code=False, interpret=False)

    @belex_apl
    def read_without_write(Belex):
        tmp = Belex.VR()
        LLB.RL[::] <= tmp()

    @belex_apl
    def read_before_write(Belex):
        tmp = Belex.VR()
        LLB.RL[::] <= tmp()
        tmp[::] <= LLB.RL()

    @belex_apl
    def read_from_unwritten_section(Belex):
        tmp = Belex.VR()
        LLB.RL[::] <= 1
        tmp["0x0002"] <= LLB.RL()
        LLB.RL["0x0003"] <= tmp()

    @belex_apl
    def read_after_init(Belex):
        tmp = Belex.VR(0)
        LLB.RL[::] <= tmp()

    @belex_apl
    def read_after_write(Belex):
        tmp = Belex.VR()
        LLB.RL[::] <= 1
        tmp[::] <= LLB.RL()
        LLB.RL[::] <= tmp()

    @belex_apl
    def read_after_param_write(Belex, msk: LLB.Mask):
        tmp = Belex.VR()
        LLB.RL[msk] <= 1
        tmp[msk] <= LLB.RL()
        LLB.RL["0x1111"] <= tmp()

    @belex_apl
    def read_after_parallel_write(Belex):
        tmp = Belex.VR()
        LLB.RL[::] <= 1
        with apl_commands():
            LLB.RL[::] <= tmp()
            tmp[::] <= LLB.RL()  # <<- will be executed before read

    # Invalid case #1
    with pytest.raises(SemanticError):
        vm.compile(read_without_write())

    # Invalid case #2
    with pytest.raises(SemanticError):
        vm.compile(read_before_write())

    # Invalid case #3
    with pytest.raises(SemanticError):
        vm.compile(read_from_unwritten_section())

    # These are valid cases
    vm.compile(read_after_init())
    vm.compile(read_after_write())
    vm.compile(read_after_param_write(0xFFFF))
    vm.compile(read_after_parallel_write())


@belex_apl
def gl_and_inv_gl_same_instr_w_same_msk(Belex, in_out: LLB.VR):
    with LLB.apl_commands():
        LLB.RL[LLB.SM_0XFFFF] <= in_out()
        LLB.GL[LLB.SM_0XFFFF] <= LLB.RL()
    with LLB.apl_commands():
        LLB.RN_REG_T0[~(LLB.SM_0XFFFF << 14)] <= LLB.INV_GL()
        LLB.RL[~(LLB.SM_0XFFFF << 14)] <= in_out() & LLB.GL()
    in_out[~(LLB.SM_0XFFFF << 14)] <= LLB.RL()


@belex_apl
def gl_and_inv_gl_same_instr_w_joint_msk(Belex, in_out: LLB.VR):
    with LLB.apl_commands():
        LLB.RL[LLB.SM_0XFFFF] <= in_out()
        LLB.GL[LLB.SM_0XFFFF] <= LLB.RL()
    with LLB.apl_commands():
        LLB.RN_REG_T0[~(LLB.SM_0X000F << 4)] <= LLB.INV_GL()
        LLB.RL[~(LLB.SM_0X000F << 5)] <= in_out() & LLB.GL()
    in_out[~(LLB.SM_0XFFFF << 14)] <= LLB.RL()


@belex_apl
def gl_and_inv_gl_same_instr_w_disj_msk(Belex, in_out: LLB.VR):
    with LLB.apl_commands():
        LLB.RL[LLB.SM_0XFFFF] <= in_out()
        LLB.GL[LLB.SM_0XFFFF] <= LLB.RL()
    with LLB.apl_commands():
        LLB.RN_REG_T0[~(LLB.SM_0XFFFF << 14)] <= LLB.INV_GL()
        LLB.RL[(LLB.SM_0XFFFF << 14)] <= in_out() & LLB.GL()
    in_out[~(LLB.SM_0XFFFF << 14)] <= LLB.RL()


@belex_apl
def gl_and_inv_gl_same_instr_w_disj_vars(Belex, in_out: LLB.VR, msk: LLB.Mask):
    with LLB.apl_commands():
        LLB.RL[LLB.SM_0XFFFF] <= in_out()
        LLB.GL[LLB.SM_0XFFFF] <= LLB.RL()
    with LLB.apl_commands():
        LLB.RN_REG_T0[~msk] <= LLB.INV_GL()
        LLB.RL[msk] <= in_out() & LLB.GL()
    in_out[~(LLB.SM_0XFFFF << 14)] <= LLB.RL()


@belex_apl
def gl_and_inv_gl_same_instr_w_joint_vars(Belex, in_out: LLB.VR, msk: LLB.Mask):
    with LLB.apl_commands():
        LLB.RL[LLB.SM_0XFFFF] <= in_out()
        LLB.GL[LLB.SM_0XFFFF] <= LLB.RL()
    with LLB.apl_commands():
        LLB.RN_REG_T0[msk] <= LLB.INV_GL()
        LLB.RL[msk << 2] <= in_out() & LLB.GL()
    in_out[~(LLB.SM_0XFFFF << 14)] <= LLB.RL()


@belex_apl
def gl_and_inv_gl_same_instr_w_joint_vars_and_msk(Belex, in_out: LLB.VR, msk: LLB.Mask):
    with LLB.apl_commands():
        LLB.RL[LLB.SM_0XFFFF] <= in_out()
        LLB.GL[LLB.SM_0XFFFF] <= LLB.RL()
    with LLB.apl_commands():
        LLB.RN_REG_T0[msk] <= LLB.INV_GL()
        LLB.RL[LLB.SM_0XFFFF] <= in_out() & LLB.GL()
    in_out[~(LLB.SM_0XFFFF << 14)] <= LLB.RL()


@belex_apl
def gl_and_inv_gl_same_instr_w_disj_vars_and_msk(Belex, in_out: LLB.VR, msk: LLB.Mask):
    with LLB.apl_commands():
        LLB.RL[LLB.SM_0XFFFF] <= in_out()
        LLB.GL[LLB.SM_0XFFFF] <= LLB.RL()
    with LLB.apl_commands():
        LLB.RN_REG_T0[msk] <= LLB.INV_GL()
        LLB.RL[LLB.SM_0X000F] <= in_out() & LLB.GL()
    in_out[~(LLB.SM_0XFFFF << 14)] <= LLB.RL()


def test_gl_and_inv_gl_same_instr():
    walker = BLEIRWalker()
    validator = MultiStatementValidator()
    in_out_vp = 0

    with pytest.raises(SemanticError):
        fcc = gl_and_inv_gl_same_instr_w_same_msk(in_out_vp)
        walker.walk(validator, fcc)

    with pytest.raises(SemanticError):
        fcc = gl_and_inv_gl_same_instr_w_joint_msk(in_out_vp)
        walker.walk(validator, fcc)

    fcc = gl_and_inv_gl_same_instr_w_disj_msk(in_out_vp)
    walker.walk(validator, fcc)

    msk_vp = 0xBEEF

    fcc = gl_and_inv_gl_same_instr_w_disj_vars(in_out_vp, msk_vp)
    walker.walk(validator, fcc)

    with pytest.raises(SemanticError):
        fcc = gl_and_inv_gl_same_instr_w_joint_vars(in_out_vp, msk_vp)
        walker.walk(validator, fcc)

    with pytest.raises(SemanticError):
        fcc = gl_and_inv_gl_same_instr_w_joint_vars_and_msk(in_out_vp, msk_vp)
        walker.walk(validator, fcc)

    msk_vp = 0x00F0

    fcc = gl_and_inv_gl_same_instr_w_disj_vars_and_msk(in_out_vp, msk_vp)
    walker.walk(validator, fcc)
