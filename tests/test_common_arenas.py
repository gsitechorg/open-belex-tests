r"""By Brian Beckman.

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
from typing import Set

import hypothesis.strategies as st
from hypothesis import given

from open_belex.common.register_arenas import (NUM_EWE_REGS, NUM_L1_REGS,
                                               NUM_L2_REGS, NUM_RE_REGS,
                                               NUM_RN_REGS, NUM_SM_REGS,
                                               EweRegArena, L1RegArena,
                                               L2RegArena, ReRegArena,
                                               RnRegArena, SmRegArena)


def test_register_arenas():
    smrs = SmRegArena()
    smr0 = smrs.allocate()
    assert smr0 == 'SM_REG_0'

    smr1 = smrs.allocate()
    assert smr1 == 'SM_REG_1'

    smrs.free(smr0)

    smr2 = smrs.allocate()
    assert smr2 == 'SM_REG_0'

    # Test double free.
    smrs.free(smr2)
    smrs_pre = deepcopy(smrs)
    smrs.free(smr0)
    smrs_post = deepcopy(smrs)
    assert smrs_post.arena == smrs_pre.arena

    # Test getting a bunch
    temp0 = smrs.allocate_several(1)
    assert temp0 == ['SM_REG_0']

    temp1 = smrs.allocate_several(3)
    assert temp1 == ['SM_REG_2', 'SM_REG_3', 'SM_REG_4']

    assert smrs.free_count() == 11

    assert smrs.allocate_several(12) is None
    assert smrs.free_count() == 11

    all_the_rest = smrs.allocate_several(10)
    assert all_the_rest is not None
    assert smrs.free_count() == 1
    assert smrs.allocated_count() == 15

    temp1 = [smrs.free(reg) for reg in all_the_rest]
    assert smrs.free_count() == 11
    assert smrs.allocated_count() == 5

    smrs.free_several(smrs.arena.keys())
    assert smrs.free_count() == 16
    assert smrs.allocated_count() == 0

    temp2 = smrs.allocate_several(8)
    assert temp2 is not None
    assert smrs.free_count() == 8
    assert smrs.allocated_count() == 8
    assert len(temp2) == smrs.allocated_count()
    smrs.free_all()
    assert smrs.free_count() == 16
    assert smrs.allocated_count() == 0


@st.composite
def sample_reservations(draw, nregs: int) -> Set[int]:
    reservations = []
    for reservation in range(nregs):
        if draw(st.booleans()):
            reservations.append(reservation)
    return set(reservations)


@given(reservations=sample_reservations(NUM_SM_REGS))
def test_reserved_sm_reg_arena(reservations: Set[int]) -> None:
    arena = SmRegArena(reservations=reservations)
    indices = set(range(NUM_SM_REGS))
    assert arena.nregs == len(indices) - len(reservations)
    for i in reservations:
        assert f"SM_REG_{i}" not in arena.arena
    for i in indices - reservations:
        assert f"SM_REG_{i}" in arena.arena


@given(reservations=sample_reservations(NUM_RN_REGS))
def test_reserved_rn_reg_arena(reservations: Set[int]) -> None:
    arena = RnRegArena(reservations=reservations)
    indices = set(range(NUM_RN_REGS))
    assert arena.nregs == len(indices) - len(reservations)
    for i in reservations:
        assert f"RN_REG_{i}" not in arena.arena
    for i in indices - reservations:
        assert f"RN_REG_{i}" in arena.arena


@given(reservations=sample_reservations(NUM_RE_REGS))
def test_reserved_re_reg_arena(reservations: Set[int]) -> None:
    arena = ReRegArena(reservations=reservations)
    indices = set(range(NUM_RE_REGS))
    assert arena.nregs == len(indices) - len(reservations)
    for i in reservations:
        assert f"RE_REG_{i}" not in arena.arena
    for i in indices - reservations:
        assert f"RE_REG_{i}" in arena.arena


@given(reservations=sample_reservations(NUM_EWE_REGS))
def test_reserved_ewe_reg_arena(reservations: Set[int]) -> None:
    arena = EweRegArena(reservations=reservations)
    indices = set(range(NUM_EWE_REGS))
    assert arena.nregs == len(indices) - len(reservations)
    for i in reservations:
        assert f"EWE_REG_{i}" not in arena.arena
    for i in indices - reservations:
        assert f"EWE_REG_{i}" in arena.arena


@given(reservations=sample_reservations(NUM_L1_REGS))
def test_reserved_l1_reg_arena(reservations: Set[int]) -> None:
    arena = L1RegArena(reservations=reservations)
    indices = set(range(NUM_L1_REGS))
    assert arena.nregs == len(indices) - len(reservations)
    for i in reservations:
        assert f"L1_ADDR_REG_{i}" not in arena.arena
    for i in indices - reservations:
        assert f"L1_ADDR_REG_{i}" in arena.arena


@given(reservations=sample_reservations(NUM_L2_REGS))
def test_reserved_l2_reg_arena(reservations: Set[int]) -> None:
    arena = L2RegArena(reservations=reservations)
    indices = set(range(NUM_L2_REGS))
    assert arena.nregs == len(indices) - len(reservations)
    for i in reservations:
        assert f"L2_ADDR_REG_{i}" not in arena.arena
    for i in indices - reservations:
        assert f"L2_ADDR_REG_{i}" in arena.arena
