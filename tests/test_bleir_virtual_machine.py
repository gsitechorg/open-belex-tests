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

from open_belex.bleir.virtual_machines import BLEIRVirtualMachine
from open_belex.common.register_arenas import NUM_RN_REGS
from open_belex.utils.config_utils import belex_config


def test_default_constructor():
    vm = BLEIRVirtualMachine()
    assert vm.reservations == {"sm_regs": set()}
    assert vm.max_rn_regs == NUM_RN_REGS


def test_setting_fields_with_params():
    reservations = {
        "sm_regs": [3, 4, 5, 6],
    }

    max_rn_regs = 8

    vm = BLEIRVirtualMachine(
        reservations=reservations,
        max_rn_regs=max_rn_regs)

    assert vm.reservations == {
        "sm_regs": {3, 4, 5, 6}
    }

    assert vm.max_rn_regs == max_rn_regs


@belex_config(
    max_rn_regs=5,
    reservations={
        "l1_rows": [8, 12, 15]
    })
def test_setting_fields_with_config():
    vm = BLEIRVirtualMachine()
    assert vm.max_rn_regs == 5
    assert vm.reservations == {
        "l1_rows": {8, 12, 15},
        "sm_regs": set(),
    }


def test_setting_max_rn_regs_with_param_and_reservations():
    reservations = {
        "rn_regs": list(range(3, NUM_RN_REGS))
    }

    vm = BLEIRVirtualMachine(
        reservations=reservations,
        max_rn_regs=6)
    assert vm.max_rn_regs == 3

    reservations = {
        "rn_regs": list(range(7, NUM_RN_REGS))
    }

    vm = BLEIRVirtualMachine(
        reservations=reservations,
        max_rn_regs=6)
    assert vm.max_rn_regs == 6


@belex_config(max_rn_regs=5)
def test_setting_max_rn_regs_with_param_and_reservations_and_config():
    vm = BLEIRVirtualMachine(max_rn_regs=6)
    assert vm.max_rn_regs == 5

    vm = BLEIRVirtualMachine(max_rn_regs=4)
    assert vm.max_rn_regs == 4

    reservations = {
        "rn_regs": list(range(3, NUM_RN_REGS))
    }

    vm = BLEIRVirtualMachine(reservations=reservations)
    assert vm.max_rn_regs == 3

    vm = BLEIRVirtualMachine(
        reservations=reservations,
        max_rn_regs=6)
    assert vm.max_rn_regs == 3

    vm = BLEIRVirtualMachine(
        reservations=reservations,
        max_rn_regs=2)
    assert vm.max_rn_regs == 2

    vm = BLEIRVirtualMachine(max_rn_regs=1)
    assert vm.max_rn_regs == 1

    reservations = {
        "rn_regs": list(range(7, NUM_RN_REGS))
    }

    vm = BLEIRVirtualMachine(reservations=reservations)
    assert vm.max_rn_regs == 5

    vm = BLEIRVirtualMachine(
        reservations=reservations,
        max_rn_regs=6)
    assert vm.max_rn_regs == 5

    vm = BLEIRVirtualMachine(
        reservations=reservations,
        max_rn_regs=2)
    assert vm.max_rn_regs == 2
