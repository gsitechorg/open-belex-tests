r"""
By Dylon Edwards
"""

from open_belex.bleir.analyzers import RegisterParameterFinder
from open_belex.bleir.types import RN_REG, FormalParameter
from open_belex.bleir.virtual_machines import BLEIRVirtualMachine
from open_belex.bleir.walkables import BLEIRWalker
from open_belex.common.constants import NUM_HALF_BANKS_PER_APUC
from open_belex.diri.half_bank import DIRI
from open_belex.literal import (INV_RL, RL, VR, WRL, Mask, apl_commands,
                                belex_apl, u16)
from open_belex.utils.example_utils import convert_to_u16

from open_belex_tests.utils import parameterized_belex_test

NUM_STEPS_TOO_MANY = 25


def is_rn_reg(formal_parameter: FormalParameter) -> bool:
    return isinstance(formal_parameter, RN_REG)


@belex_apl
def frag_w_too_many_temps(Belex, out: VR, value: u16):
    with apl_commands():
        RL[value] <= 1
        RL[~value] <= 0
    for step in range(NUM_STEPS_TOO_MANY):
        temp = Belex.VR(0)
        temp[::] <= WRL()
        RL[::] <= temp()
        out[::] <= RL()


@parameterized_belex_test
def test_frag_w_too_many_temps(diri: DIRI) -> int:
    out = 0
    value = 0x1234

    vm = BLEIRVirtualMachine(interpret=False, generate_code=False)
    call = vm.compile(frag_w_too_many_temps(out, value))
    fragment = call.fragment

    finder = RegisterParameterFinder()
    walker = BLEIRWalker()
    walker.walk(finder, fragment)

    # One user param and two temporaries (one for initialization)
    assert len(list(filter(is_rn_reg, finder.register_parameters))) == 1
    assert len(list(filter(is_rn_reg, finder.lowered_registers))) == 1

    for hb in range(NUM_HALF_BANKS_PER_APUC):
        vr = diri[hb, out, ::, ::]
        assert all(convert_to_u16(vr[:NUM_STEPS_TOO_MANY]) == 0x0000)
        assert all(convert_to_u16(vr[NUM_STEPS_TOO_MANY:]) == value)

    return out


@belex_apl
def frag_w_parallel_non_overlapping_temps(Belex, out: VR, value: u16):
    lower_half = Belex.VR()
    upper_half = Belex.VR()
    with apl_commands():
        RL[value] <= 1
        RL[~value] <= 0
    lower_half[:8] <= RL()
    upper_half[8:] <= RL()
    RL[:8] <= lower_half()
    RL[8:] |= upper_half()
    out[::] <= RL()


@parameterized_belex_test
def test_frag_w_parallel_non_overlapping_temps(diri: DIRI) -> int:
    out = 0
    value = 0xBEEF

    vm = BLEIRVirtualMachine(interpret=False, generate_code=False)
    call = vm.compile(frag_w_parallel_non_overlapping_temps(out, value))
    fragment = call.fragment

    finder = RegisterParameterFinder()
    walker = BLEIRWalker()
    walker.walk(finder, fragment)

    # One user param and one temporary
    assert len(list(filter(is_rn_reg, finder.register_parameters))) == 1
    assert len(list(filter(is_rn_reg, finder.lowered_registers))) == 1

    assert all(convert_to_u16(diri.hb[out]) == value)
    return out


@belex_apl
def frag_w_too_many_temps_w_mask_param(Belex, out: VR, value: Mask):
    for step in range(NUM_STEPS_TOO_MANY):
        upper_half = Belex.VR()
        lower_half = Belex.VR()
        RL[::] <= 1
        with apl_commands():
            upper_half[value] <= RL()
            lower_half[~value] <= INV_RL()
        with apl_commands():
            RL[value] <= upper_half()
            RL[~value] <= lower_half()
        out[::] <= RL()


@parameterized_belex_test
def test_frag_w_too_many_temps_w_mask_param(diri: DIRI) -> int:
    out = 0
    value = 0x1234

    vm = BLEIRVirtualMachine(interpret=False, generate_code=False)
    call = vm.compile(frag_w_too_many_temps_w_mask_param(out, value))
    fragment = call.fragment

    finder = RegisterParameterFinder()
    walker = BLEIRWalker()
    walker.walk(finder, fragment)

    # One user param and one temporary
    assert len(list(filter(is_rn_reg, finder.register_parameters))) == 1
    assert len(list(filter(is_rn_reg, finder.lowered_registers))) == 1

    assert all(convert_to_u16(diri.hb[out]) == value)
    return out
