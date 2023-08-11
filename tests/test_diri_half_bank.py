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
from itertools import chain
from typing import Any, Callable, Sequence

import numpy as np

import pytest

import hypothesis
import hypothesis.strategies as st
from hypothesis import given

from open_belex.bleir.virtual_machines import BLEIRVirtualMachine
from open_belex.common.constants import (NPLATS, NSB, NSECTIONS,
                                         NUM_HALF_BANKS_PER_APUC, NUM_L2_ROWS,
                                         NUM_PLATS_PER_APUC,
                                         NUM_PLATS_PER_HALF_BANK)
from open_belex.common.mask import Mask
from open_belex.common.rsp_fifo import FIFO_CAPACITY, ApucRspFifo
from open_belex.diri.half_bank import (DIRI, GGL_SHAPE, VALID_L1_ADDRESSES,
                                       plats_for_bank)
from open_belex.kernel_libs.memory import NUM_VM_REGS
from open_belex.literal import (NOOP, RL, RSP2K, RSP16, RSP32K, RSP256,
                                RSP_END, VR, belex_apl)
from open_belex.utils.example_utils import convert_to_bool

from open_belex_libs.memory import store_16

from open_belex_tests.utils import (Mask_strategy, ggl_strategy, gl_strategy,
                                    lgl_strategy, rsp2k_strategy,
                                    rsp16_strategy, rsp32k_strategy,
                                    rsp256_strategy, vr_strategy)

section_strategy = st.integers(0, NSECTIONS - 1)
sb_strategy = st.integers(0, NSB - 1)
sbs_strategy = st.lists(sb_strategy, min_size=1, max_size=3)
vrs_strategy = st.lists(vr_strategy(), min_size=1, max_size=3)
binary_strategy = st.integers(0, 1)
l1_addr_strategy = st.sampled_from(np.where(VALID_L1_ADDRESSES)[0])
l2_addr_strategy = st.integers(0, NUM_L2_ROWS - 1)
vmr_strategy = st.integers(0, NUM_VM_REGS - 1)


def diri_test(fn: Callable = None,
              in_place: bool = True,
              repeatably_randomize_half_bank: bool = False) -> Callable:

    def decorator(fn: Callable) -> Callable:

        def wrapper(*args, **kwargs) -> Any:
            diri = DIRI(in_place=in_place)
            if repeatably_randomize_half_bank:
                diri.repeatably_randomize_half_bank()
            return fn(diri, *args, **kwargs)

        wrapper.__fn__ = fn
        wrapper.__name__ = fn.__name__
        return wrapper

    if fn is not None:
        return decorator(fn)

    return decorator


def diri_property_test(fn: Callable = None,
                       in_place: bool = True,
                       repeatably_randomize_half_bank: bool = False,
                       max_examples: int = 5,
                       **kwargs) -> Callable:

    def decorator(fn: Callable) -> Callable:

        @hypothesis.settings(deadline=None,
                             max_examples=max_examples,
                             print_blob=True,
                             # Options: quiet, normal, verbose, debug
                             verbosity=hypothesis.Verbosity.normal,
                             report_multiple_bugs=False,
                             database=None,  # no storage
                             phases=(
                                 # hypothesis.Phase.explicit,
                                 # hypothesis.Phase.reuse,
                                 hypothesis.Phase.generate,
                                 # hypothesis.Phase.target,
                                 # hypothesis.Phase.shrink
                             ),
                             suppress_health_check=[
                                 hypothesis.HealthCheck.data_too_large,
                                 hypothesis.HealthCheck.large_base_example,
                                 hypothesis.HealthCheck.too_slow,
                             ])
        @given(**kwargs)
        @diri_test(in_place=in_place,
                   repeatably_randomize_half_bank=repeatably_randomize_half_bank)
        def wrapper(diri: DIRI, *args, **kwargs) -> Any:
            return fn(diri, *args, **kwargs)

        wrapper.__fn__ = fn
        wrapper.__name__ = fn.__name__
        return wrapper

    return decorator


@diri_property_test(vr=vr_strategy(), sb=sb_strategy)
def test_reset_sb(diri: DIRI, vr: np.ndarray, sb: int) -> None:
    diri.hb[sb, ::, ::] = convert_to_bool(vr)
    diri.reset_sb(sb)
    assert not diri.hb[sb].any()


@diri_property_test(vr=vr_strategy(), sb=sb_strategy, section=section_strategy)
def test_reset_sb_section(diri: DIRI, vr: np.ndarray, sb: int, section: int) -> None:
    diri.hb[sb, ::, ::] = convert_to_bool(vr)
    diri.reset_sb_section(section, sb)
    assert not diri.hb[sb, ::, section].any()


@diri_property_test(rl=vr_strategy())
def test_reset_rl(diri: DIRI, rl: np.ndarray) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.reset_rl()
    assert not diri.RL().any()


@diri_property_test(rl=vr_strategy(), section=section_strategy)
def test_reset_rl_section(diri: DIRI, rl: np.ndarray, section: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.reset_rl_section(section)
    assert not diri.RL()[::, section].any()


@diri_property_test(gl=gl_strategy())
def test_reset_gl(diri: DIRI, gl: np.ndarray) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    diri.reset_gl()
    assert not diri.GL.any()


@diri_property_test(ggl=ggl_strategy())
def test_reset_ggl(diri: DIRI, ggl: np.ndarray) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    diri.reset_ggl()
    assert not diri.GGL.any()


@diri_property_test(rsp16=rsp16_strategy())
def test_reset_rsp16(diri: DIRI, rsp16: np.ndarray) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    diri.reset_rsp16()
    assert not diri.RSP16.any()


@diri_property_test(rsp256=rsp256_strategy())
def test_reset_rsp256(diri: DIRI, rsp256: np.ndarray) -> None:
    diri.RSP256[::, ::] = convert_to_bool(rsp256)
    diri.reset_rsp256()
    assert not diri.RSP256.any()


@diri_property_test(rsp2k=rsp2k_strategy())
def test_reset_rsp2k(diri: DIRI, rsp2k: np.ndarray) -> None:
    diri.RSP2K[::, ::] = convert_to_bool(rsp2k)
    diri.reset_rsp2k()
    assert not diri.RSP2K.any()


@diri_property_test(rsp32k=rsp32k_strategy())
def test_reset_rsp32k(diri: DIRI, rsp32k: np.ndarray) -> None:
    diri.RSP32K[::, ::] = convert_to_bool(rsp32k)
    diri.reset_rsp32k()
    assert not diri.RSP32K.any()


@diri_property_test(vmr=vr_strategy())
def test_reset_l1(diri: DIRI, vmr: np.ndarray) -> None:
    # Full L1 randomization takes a LONG time ...
    diri.L1[0:4, ::, ::] = convert_to_bool(vmr) \
        .reshape((4, NUM_PLATS_PER_APUC, 4))
    diri.reset_l1()
    assert not diri.L1.any()


@diri_property_test(l2=lgl_strategy())
def test_reset_l2(diri: DIRI, l2: np.ndarray) -> None:
    # Full L2 randomization takes a LONG time ...
    diri.L2[0] = convert_to_bool(l2, nsections=1)[::, 0]
    diri.reset_l2()
    assert not diri.L2.any()


@diri_property_test(lgl=lgl_strategy())
def test_reset_lgl(diri: DIRI, lgl: np.ndarray) -> None:
    diri.LGL[::, None] = convert_to_bool(lgl, nsections=1)
    diri.reset_lgl()
    assert not diri.LGL.any()


@diri_property_test(vr=vr_strategy(), sections=Mask_strategy())
def test_pull_rsps(diri: DIRI, vr: np.ndarray, sections: int) -> None:
    mask = Mask(f"0x{sections:04X}")
    diri.RL()[::, ::] = convert_to_bool(vr)
    diri.pull_rsps(mask)
    expected_value = np.zeros((1, NSECTIONS), dtype=bool)
    ss = list(mask)
    for half_bank in range(NUM_HALF_BANKS_PER_APUC):
        lower_plat = half_bank * NUM_PLATS_PER_HALF_BANK
        upper_plat = lower_plat + NUM_PLATS_PER_HALF_BANK
        expected_value[0, half_bank] = \
            diri.RL()[lower_plat:upper_plat, ss].any()
    assert np.array_equal(expected_value, diri.RSP32K)


@diri_test(repeatably_randomize_half_bank=True)
def test_noop(diri: DIRI):
    orig = deepcopy(diri)
    diri.noop()
    assert diri == orig


@diri_test(repeatably_randomize_half_bank=True)
def test_rsp_end(diri: DIRI):
    orig = deepcopy(diri)
    diri.rsp_end()
    assert diri == orig


@diri_test(repeatably_randomize_half_bank=True)
def test_rsp_start_ret(diri: DIRI):
    orig = deepcopy(diri)
    diri.rsp_start_ret()
    assert diri == orig


# ===========
# sb_from_src
# ===========


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_sbs_from_rl(diri: DIRI,
                     sbs: Sequence[int],
                     rl: np.ndarray,
                     sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = diri.RL()[::, section]
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_from_src(mask, sbs, diri.RL())
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_vrs_from_rl(diri: DIRI,
                     vrs: Sequence[np.ndarray],
                     rl: np.ndarray,
                     sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = diri.RL()[::, section]
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_from_src(mask, vrs, diri.RL())
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_sbs_from_gl(diri: DIRI,
                     sbs: Sequence[int],
                     gl: np.ndarray,
                     sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = diri.GL
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_from_src(mask, sbs, diri.GL)
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_vrs_from_gl(diri: DIRI,
                     vrs: Sequence[np.ndarray],
                     gl: np.ndarray,
                     sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = diri.GL
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_from_src(mask, vrs, diri.GL)
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_sbs_from_ggl(diri: DIRI,
                      sbs: Sequence[int],
                      ggl: np.ndarray,
                      sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = diri.GGL[::, section // 4]
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_from_src(mask, sbs, diri.GGL)
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_vrs_from_ggl(diri: DIRI,
                      vrs: Sequence[np.ndarray],
                      ggl: np.ndarray,
                      sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = diri.GGL[::, section // 4]
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_from_src(mask, vrs, diri.GGL)
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_sbs_from_rsp16(diri: DIRI,
                        sbs: Sequence[int],
                        rsp16: np.ndarray,
                        sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            for index in range(NUM_PLATS_PER_APUC // 16):
                start = index * 16
                stop = start + 16
                value = diri.RSP16[index, section]
                expected_value[start:stop, section] = value
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_from_src(mask, sbs, diri.RSP16)
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_vrs_from_rsp16(diri: DIRI,
                        vrs: Sequence[np.ndarray],
                        rsp16: np.ndarray,
                        sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            for index in range(NUM_PLATS_PER_APUC // 16):
                start = index * 16
                stop = start + 16
                value = diri.RSP16[index, section]
                expected_value[start:stop, section] = value
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_from_src(mask, vrs, diri.RSP16)
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


# ===============
# sb_from_inv_src
# ===============


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_sbs_from_inv_rl(diri: DIRI,
                         sbs: Sequence[int],
                         rl: np.ndarray,
                         sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = ~diri.RL()[::, section]
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_from_inv_src(mask, sbs, diri.RL())
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_vrs_from_inv_rl(diri: DIRI,
                         vrs: Sequence[np.ndarray],
                         rl: np.ndarray,
                         sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = ~diri.RL()[::, section]
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_from_inv_src(mask, vrs, diri.RL())
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_sbs_from_inv_gl(diri: DIRI,
                         sbs: Sequence[int],
                         gl: np.ndarray,
                         sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = ~diri.GL
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_from_inv_src(mask, sbs, diri.GL)
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_vrs_from_inv_gl(diri: DIRI,
                         vrs: Sequence[np.ndarray],
                         gl: np.ndarray,
                         sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = ~diri.GL
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_from_inv_src(mask, vrs, diri.GL)
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_sbs_from_inv_ggl(diri: DIRI,
                          sbs: Sequence[int],
                          ggl: np.ndarray,
                          sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = ~diri.GGL[::, section // 4]
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_from_inv_src(mask, sbs, diri.GGL)
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_vrs_from_inv_ggl(diri: DIRI,
                          vrs: Sequence[np.ndarray],
                          ggl: np.ndarray,
                          sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = ~diri.GGL[::, section // 4]
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_from_inv_src(mask, vrs, diri.GGL)
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_sbs_from_inv_rsp16(diri: DIRI,
                            sbs: Sequence[int],
                            rsp16: np.ndarray,
                            sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            for index in range(NUM_PLATS_PER_APUC // 16):
                start = index * 16
                stop = start + 16
                value = ~diri.RSP16[index, section]
                expected_value[start:stop, section] = value
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_from_inv_src(mask, sbs, diri.RSP16)
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_vrs_from_inv_rsp16(diri: DIRI,
                            vrs: Sequence[np.ndarray],
                            rsp16: np.ndarray,
                            sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            for index in range(NUM_PLATS_PER_APUC // 16):
                start = index * 16
                stop = start + 16
                value = ~diri.RSP16[index, section]
                expected_value[start:stop, section] = value
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_from_inv_src(mask, vrs, diri.RSP16)
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


# ==================
# sb_cond_equals_src
# ==================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_sbs_cond_equals_rl(diri: DIRI,
                            sbs: Sequence[int],
                            rl: np.ndarray,
                            sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = \
                diri.hb[sb, ::, section] | diri.RL()[::, section]
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_src(mask, sbs, diri.RL())
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_vrs_cond_equals_rl(diri: DIRI,
                            vrs: Sequence[np.ndarray],
                            rl: np.ndarray,
                            sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = \
                vr[::, section] | diri.RL()[::, section]
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_src(mask, vrs, diri.RL())
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_sbs_cond_equals_gl(diri: DIRI,
                            sbs: Sequence[int],
                            gl: np.ndarray,
                            sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = \
                diri.hb[sb, ::, section] | diri.GL
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_src(mask, sbs, diri.GL)
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_vrs_cond_equals_gl(diri: DIRI,
                            vrs: Sequence[np.ndarray],
                            gl: np.ndarray,
                            sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = \
                vr[::, section] | diri.GL
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_src(mask, vrs, diri.GL)
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_sbs_cond_equals_ggl(diri: DIRI,
                             sbs: Sequence[int],
                             ggl: np.ndarray,
                             sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = \
                diri.hb[sb, ::, section] | diri.GGL[::, section // 4]
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_src(mask, sbs, diri.GGL)
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_vrs_cond_equals_ggl(diri: DIRI,
                             vrs: Sequence[np.ndarray],
                             ggl: np.ndarray,
                             sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = \
                vr[::, section] | diri.GGL[::, section // 4]
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_src(mask, vrs, diri.GGL)
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_sbs_cond_equals_rsp16(diri: DIRI,
                               sbs: Sequence[int],
                               rsp16: np.ndarray,
                               sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            for index in range(NUM_PLATS_PER_APUC // 16):
                start = index * 16
                stop = start + 16
                value = diri.RSP16[index, section]
                expected_value[start:stop, section] = \
                    diri.hb[sb, start:stop, section] | value
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_src(mask, sbs, diri.RSP16)
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_vrs_cond_equals_rsp16(diri: DIRI,
                               vrs: Sequence[np.ndarray],
                               rsp16: np.ndarray,
                               sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            for index in range(NUM_PLATS_PER_APUC // 16):
                start = index * 16
                stop = start + 16
                value = diri.RSP16[index, section]
                expected_value[start:stop, section] = \
                    vr[start:stop, section] | value
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_src(mask, vrs, diri.RSP16)
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


# ======================
# sb_cond_equals_inv_src
# ======================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_sbs_cond_equals_inv_rl(diri: DIRI,
                                sbs: Sequence[int],
                                rl: np.ndarray,
                                sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = \
                diri.hb[sb, ::, section] & ~diri.RL()[::, section]
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_inv_src(mask, sbs, diri.RL())
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_vrs_cond_equals_inv_rl(diri: DIRI,
                                vrs: Sequence[np.ndarray],
                                rl: np.ndarray,
                                sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = \
                vr[::, section] & ~diri.RL()[::, section]
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_inv_src(mask, vrs, diri.RL())
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_sbs_cond_equals_inv_gl(diri: DIRI,
                                sbs: Sequence[int],
                                gl: np.ndarray,
                                sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = \
                diri.hb[sb, ::, section] & ~diri.GL
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_inv_src(mask, sbs, diri.GL)
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_vrs_cond_equals_inv_gl(diri: DIRI,
                                vrs: Sequence[np.ndarray],
                                gl: np.ndarray,
                                sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = \
                vr[::, section] & ~diri.GL
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_inv_src(mask, vrs, diri.GL)
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_sbs_cond_equals_inv_ggl(diri: DIRI,
                                 sbs: Sequence[int],
                                 ggl: np.ndarray,
                                 sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = \
                diri.hb[sb, ::, section] & ~diri.GGL[::, section // 4]
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_inv_src(mask, sbs, diri.GGL)
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_vrs_cond_equals_inv_ggl(diri: DIRI,
                                 vrs: Sequence[np.ndarray],
                                 ggl: np.ndarray,
                                 sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            expected_value[::, section] = \
                vr[::, section] & ~diri.GGL[::, section // 4]
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_inv_src(mask, vrs, diri.GGL)
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_sbs_cond_equals_inv_rsp16(diri: DIRI,
                                   sbs: Sequence[int],
                                   rsp16: np.ndarray,
                                   sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_values = []
    for sb in sbs:
        expected_value = diri.build_vr()
        for section in list(mask):
            for index in range(NUM_PLATS_PER_APUC // 16):
                start = index * 16
                stop = start + 16
                value = ~diri.RSP16[index, section]
                expected_value[start:stop, section] = \
                    diri.hb[sb, start:stop, section] & value
        for section in list(~mask):
            expected_value[::, section] = diri.hb[sb, ::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_inv_src(mask, sbs, diri.RSP16)
    for sb, expected_value in zip(sbs, expected_values):
        assert np.array_equal(diri.hb[sb], expected_value)


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_vrs_cond_equals_inv_rsp16(diri: DIRI,
                                   vrs: Sequence[np.ndarray],
                                   rsp16: np.ndarray,
                                   sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    orig = deepcopy(diri)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_values = []
    for vr in vrs:
        expected_value = diri.build_vr()
        for section in list(mask):
            for index in range(NUM_PLATS_PER_APUC // 16):
                start = index * 16
                stop = start + 16
                value = ~diri.RSP16[index, section]
                expected_value[start:stop, section] = \
                    vr[start:stop, section] & value
        for section in list(~mask):
            expected_value[::, section] = vr[::, section]
        expected_values.append(expected_value)
    diri.sb_cond_equals_inv_src(mask, vrs, diri.RSP16)
    assert orig == diri  # assert no internal state change
    for vr, expected_value in zip(vrs, expected_values):
        assert np.array_equal(vr, expected_value)


# ======
# set_rl
# ======


@diri_property_test(rl=vr_strategy(),
                    sections=Mask_strategy(),
                    bit=binary_strategy)
def test_set_rl(diri: DIRI, rl: np.ndarray, sections: int, bit: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    secs, inv_secs = list(mask), list(~mask)
    expected_value = diri.build_vr()
    expected_value[::, secs] = (1 == bit)
    expected_value[::, inv_secs] = diri.RL()[::, inv_secs]
    diri.set_rl(mask, bit)
    assert np.array_equal(expected_value, diri.RL())


# ===========
# rl_from_src
# ===========


@diri_property_test(rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_rl(diri: DIRI, rl: np.ndarray, sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = deepcopy(diri.RL())
    diri.rl_from_src(mask, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_gl(diri: DIRI,
                    rl: np.ndarray,
                    gl: np.ndarray,
                    sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_src(mask, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_ggl(diri: DIRI,
                     rl: np.ndarray,
                     ggl: np.ndarray,
                     sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_src(mask, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_rsp16(diri: DIRI,
                       rl: np.ndarray,
                       rsp16: np.ndarray,
                       sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] = value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_src(mask, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# ===============
# rl_from_inv_src
# ===============


@diri_property_test(rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_rl(diri: DIRI, rl: np.ndarray, sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = ~diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_src(mask, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_gl(diri: DIRI,
                        rl: np.ndarray,
                        gl: np.ndarray,
                        sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = ~diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_src(mask, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_ggl(diri: DIRI,
                         rl: np.ndarray,
                         ggl: np.ndarray,
                         sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = ~diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_src(mask, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_rsp16(diri: DIRI,
                           rl: np.ndarray,
                           rsp16: np.ndarray,
                           sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] = ~value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_src(mask, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# ================
# rl_or_equals_src
# ================


@diri_property_test(rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_or_equals_rl(diri: DIRI, rl: np.ndarray, sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = deepcopy(diri.RL())
    diri.rl_or_equals_src(mask, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_or_equals_gl(diri: DIRI,
                         rl: np.ndarray,
                         gl: np.ndarray,
                         sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = diri.RL()[::, section] | diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_or_equals_src(mask, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_or_equals_ggl(diri: DIRI,
                          rl: np.ndarray,
                          ggl: np.ndarray,
                          sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = \
            diri.RL()[::, section] | diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_or_equals_src(mask, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_or_equals_rsp16(diri: DIRI,
                            rl: np.ndarray,
                            rsp16: np.ndarray,
                            sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] = \
                diri.RL()[start:stop, section] | value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_or_equals_src(mask, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# =================
# rl_and_equals_src
# =================


@diri_property_test(rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_rl(diri: DIRI, rl: np.ndarray, sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = deepcopy(diri.RL())
    diri.rl_and_equals_src(mask, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_gl(diri: DIRI,
                          rl: np.ndarray,
                          gl: np.ndarray,
                          sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = diri.RL()[::, section] & diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_src(mask, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_ggl(diri: DIRI,
                           rl: np.ndarray,
                           ggl: np.ndarray,
                           sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = \
            diri.RL()[::, section] & diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_src(mask, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_rsp16(diri: DIRI,
                             rl: np.ndarray,
                             rsp16: np.ndarray,
                             sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] = \
                diri.RL()[start:stop, section] & value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_src(mask, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# =====================
# rl_and_equals_inv_src
# =====================


@diri_property_test(rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_inv_rl(diri: DIRI,
                              rl: np.ndarray,
                              sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = False
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_inv_src(mask, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_inv_gl(diri: DIRI,
                              rl: np.ndarray,
                              gl: np.ndarray,
                              sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = diri.RL()[::, section] & ~diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_inv_src(mask, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_inv_ggl(diri: DIRI,
                               rl: np.ndarray,
                               ggl: np.ndarray,
                               sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = \
            diri.RL()[::, section] & ~diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_inv_src(mask, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_inv_rsp16(diri: DIRI,
                                 rl: np.ndarray,
                                 rsp16: np.ndarray,
                                 sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] = \
                diri.RL()[start:stop, section] & ~value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_inv_src(mask, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# =================
# rl_xor_equals_src
# =================


@diri_property_test(rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_rl(diri: DIRI, rl: np.ndarray, sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = False
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_src(mask, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_gl(diri: DIRI,
                          rl: np.ndarray,
                          gl: np.ndarray,
                          sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = diri.RL()[::, section] ^ diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_src(mask, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_ggl(diri: DIRI,
                           rl: np.ndarray,
                           ggl: np.ndarray,
                           sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = \
            diri.RL()[::, section] ^ diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_src(mask, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_rsp16(diri: DIRI,
                             rl: np.ndarray,
                             rsp16: np.ndarray,
                             sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] = \
                diri.RL()[start:stop, section] ^ value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_src(mask, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# =====================
# rl_xor_equals_inv_src
# =====================


@diri_property_test(rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_inv_rl(diri: DIRI,
                              rl: np.ndarray,
                              sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = True
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_inv_src(mask, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_inv_gl(diri: DIRI,
                              rl: np.ndarray,
                              gl: np.ndarray,
                              sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = diri.RL()[::, section] ^ ~diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_inv_src(mask, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_inv_ggl(diri: DIRI,
                               rl: np.ndarray,
                               ggl: np.ndarray,
                               sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        expected_value[::, section] = \
            diri.RL()[::, section] ^ ~diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_inv_src(mask, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(rl=vr_strategy(),
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_inv_rsp16(diri: DIRI,
                                 rl: np.ndarray,
                                 rsp16: np.ndarray,
                                 sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] = \
                diri.RL()[start:stop, section] ^ ~value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_inv_src(mask, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# ==========
# rl_from_sb
# ==========


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs(diri: DIRI,
                     sbs: Sequence[int],
                     rl: np.ndarray,
                     sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb(mask, sbs)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs(diri: DIRI,
                     vrs: Sequence[np.ndarray],
                     rl: np.ndarray,
                     sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb(mask, vrs)
    assert np.array_equal(expected_value, diri.RL())


# ==============
# rl_from_inv_sb
# ==============


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_sbs(diri: DIRI,
                         sbs: Sequence[int],
                         rl: np.ndarray,
                         sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            # By De Morgan's Law, ~(a AND b) == (~a) OR (~b)
            if index == 0:
                expected_value[::, section] = ~diri.hb[sb, ::, section]
            else:
                expected_value[::, section] |= ~diri.hb[sb, ::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb(mask, sbs)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_vrs(diri: DIRI,
                         vrs: Sequence[np.ndarray],
                         rl: np.ndarray,
                         sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            # By De Morgan's Law, ~(a AND b) == (~a) OR (~b)
            if index == 0:
                expected_value[::, section] = ~vr[::, section]
            else:
                expected_value[::, section] |= ~vr[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb(mask, vrs)
    assert np.array_equal(expected_value, diri.RL())


# ===============
# rl_or_equals_sb
# ===============


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_or_equals_sbs(diri: DIRI,
                          sbs: Sequence[int],
                          rl: np.ndarray,
                          sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = deepcopy(diri.RL())
    if sections > 0x0000:
        secs = list(mask)
        grid = np.ix_(sbs, list(range(NUM_PLATS_PER_APUC)), secs)
        expected_value[::, secs] |= \
            np.logical_and.reduce(diri.hb[grid], axis=0)
    diri.rl_or_equals_sb(mask, sbs)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_or_equals_vrs(diri: DIRI,
                          vrs: Sequence[np.ndarray],
                          rl: np.ndarray,
                          sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = deepcopy(diri.RL())
    if sections > 0x0000:
        secs = list(mask)
        grid = np.ix_(list(range(NUM_PLATS_PER_APUC)), secs)
        expected_value[::, secs] |= \
            np.logical_and.reduce([vr[grid] for vr in vrs], axis=0)
    diri.rl_or_equals_sb(mask, vrs)
    assert np.array_equal(expected_value, diri.RL())


# ================
# rl_and_equals_sb
# ================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_sbs(diri: DIRI,
                           sbs: Sequence[int],
                           rl: np.ndarray,
                           sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = deepcopy(diri.RL())
    for sb in sbs:
        for section in list(mask):
            expected_value[::, section] &= diri.hb[sb, ::, section]
    diri.rl_and_equals_sb(mask, sbs)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_vrs(diri: DIRI,
                           vrs: Sequence[np.ndarray],
                           rl: np.ndarray,
                           sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = deepcopy(diri.RL())
    for index, vr in enumerate(vrs):
        for section in list(mask):
            expected_value[::, section] &= vr[::, section]
    diri.rl_and_equals_sb(mask, vrs)
    assert np.array_equal(expected_value, diri.RL())


# ====================
# rl_and_equals_inv_sb
# ====================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_inv_sbs(diri: DIRI,
                               sbs: Sequence[int],
                               rl: np.ndarray,
                               sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for sb in sbs:
        for section in list(mask):
            # By De Morgan's Law, A & ~(B & C) == A & (~B | ~C) == (A & ~B) | (A & ~C)
            expected_value[::, section] |= \
                diri.RL()[::, section] & ~diri.hb[sb, ::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_inv_sb(mask, sbs)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_inv_vrs(diri: DIRI,
                               vrs: Sequence[np.ndarray],
                               rl: np.ndarray,
                               sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            expected_value[::, section] |= \
                diri.RL()[::, section] & ~vr[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_inv_sb(mask, vrs)
    assert np.array_equal(expected_value, diri.RL())


# ================
# rl_xor_equals_sb
# ================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_sbs(diri: DIRI,
                           sbs: Sequence[int],
                           rl: np.ndarray,
                           sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] ^= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_sb(mask, sbs)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_vrs(diri: DIRI,
                           vrs: Sequence[np.ndarray],
                           rl: np.ndarray,
                           sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] ^= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_sb(mask, vrs)
    assert np.array_equal(expected_value, diri.RL())


# ==================
# rl_from_sb_and_src
# ==================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_and_rl(diri: DIRI,
                            sbs: Sequence[int],
                            rl: np.ndarray,
                            sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_src(mask, sbs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_and_rl(diri: DIRI,
                            vrs: Sequence[np.ndarray],
                            rl: np.ndarray,
                            sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_src(mask, vrs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_and_gl(diri: DIRI,
                            sbs: Sequence[int],
                            gl: np.ndarray,
                            sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_src(mask, sbs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_and_gl(diri: DIRI,
                            vrs: Sequence[np.ndarray],
                            gl: np.ndarray,
                            sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_src(mask, vrs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_and_ggl(diri: DIRI,
                             sbs: Sequence[int],
                             ggl: np.ndarray,
                             sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_src(mask, sbs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_and_ggl(diri: DIRI,
                             vrs: Sequence[np.ndarray],
                             ggl: np.ndarray,
                             sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_src(mask, vrs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_and_rsp16(diri: DIRI,
                               sbs: Sequence[int],
                               rsp16: np.ndarray,
                               sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] &= value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_src(mask, sbs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_and_rsp16(diri: DIRI,
                               vrs: Sequence[np.ndarray],
                               rsp16: np.ndarray,
                               sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] &= value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_src(mask, vrs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# =======================
# rl_or_equals_sb_and_src
# =======================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_or_equals_sbs_and_rl(diri: DIRI,
                                 sbs: Sequence[int],
                                 rl: np.ndarray,
                                 sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.RL()[::, section]
        expected_value[::, section] |= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_or_equals_sb_and_src(mask, sbs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_or_equals_vrs_and_rl(diri: DIRI,
                                 vrs: Sequence[np.ndarray],
                                 rl: np.ndarray,
                                 sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.RL()[::, section]
        expected_value[::, section] |= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_or_equals_sb_and_src(mask, vrs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_or_equals_sbs_and_gl(diri: DIRI,
                                 sbs: Sequence[int],
                                 gl: np.ndarray,
                                 sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GL
        expected_value[::, section] |= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_or_equals_sb_and_src(mask, sbs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_or_equals_vrs_and_gl(diri: DIRI,
                                 vrs: Sequence[np.ndarray],
                                 gl: np.ndarray,
                                 sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GL
        expected_value[::, section] |= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_or_equals_sb_and_src(mask, vrs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_or_equals_sbs_and_ggl(diri: DIRI,
                                 sbs: Sequence[int],
                                 ggl: np.ndarray,
                                 sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GGL[::, section // 4]
        expected_value[::, section] |= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_or_equals_sb_and_src(mask, sbs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_or_equals_vrs_and_ggl(diri: DIRI,
                                  vrs: Sequence[np.ndarray],
                                  ggl: np.ndarray,
                                  sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GGL[::, section // 4]
        expected_value[::, section] |= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_or_equals_sb_and_src(mask, vrs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_or_equals_sbs_and_rsp16(diri: DIRI,
                                    sbs: Sequence[int],
                                    rsp16: np.ndarray,
                                    sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] &= value
        expected_value[::, section] |= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_or_equals_sb_and_src(mask, sbs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_or_equals_vrs_and_rsp16(diri: DIRI,
                                    vrs: Sequence[np.ndarray],
                                    rsp16: np.ndarray,
                                    sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] &= value
        expected_value[::, section] |= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_or_equals_sb_and_src(mask, vrs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# ========================
# rl_and_equals_sb_and_src
# ========================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_sbs_and_rl(diri: DIRI,
                                  sbs: Sequence[int],
                                  rl: np.ndarray,
                                  sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_sb_and_src(mask, sbs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_vrs_and_rl(diri: DIRI,
                                  vrs: Sequence[np.ndarray],
                                  rl: np.ndarray,
                                  sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_sb_and_src(mask, vrs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_sbs_and_gl(diri: DIRI,
                                  sbs: Sequence[int],
                                  gl: np.ndarray,
                                  sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GL
        expected_value[::, section] &= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_sb_and_src(mask, sbs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_vrs_and_gl(diri: DIRI,
                                  vrs: Sequence[np.ndarray],
                                  gl: np.ndarray,
                                  sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GL
        expected_value[::, section] &= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_sb_and_src(mask, vrs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_sbs_and_ggl(diri: DIRI,
                                   sbs: Sequence[int],
                                   ggl: np.ndarray,
                                   sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GGL[::, section // 4]
        expected_value[::, section] &= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_sb_and_src(mask, sbs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_vrs_and_ggl(diri: DIRI,
                                   vrs: Sequence[np.ndarray],
                                   ggl: np.ndarray,
                                   sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GGL[::, section // 4]
        expected_value[::, section] &= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_sb_and_src(mask, vrs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_sbs_and_rsp16(diri: DIRI,
                                     sbs: Sequence[int],
                                     rsp16: np.ndarray,
                                     sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] &= value
        expected_value[::, section] &= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_sb_and_src(mask, sbs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_and_equals_vrs_and_rsp16(diri: DIRI,
                                     vrs: Sequence[np.ndarray],
                                     rsp16: np.ndarray,
                                     sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] &= value
        expected_value[::, section] &= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_and_equals_sb_and_src(mask, vrs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# ========================
# rl_xor_equals_sb_and_src
# ========================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_sbs_and_rl(diri: DIRI,
                                  sbs: Sequence[int],
                                  rl: np.ndarray,
                                  sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.RL()[::, section]
        expected_value[::, section] ^= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_sb_and_src(mask, sbs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_vrs_and_rl(diri: DIRI,
                                  vrs: Sequence[np.ndarray],
                                  rl: np.ndarray,
                                  sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.RL()[::, section]
        expected_value[::, section] ^= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_sb_and_src(mask, vrs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_sbs_and_gl(diri: DIRI,
                                  sbs: Sequence[int],
                                  gl: np.ndarray,
                                  sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GL
        expected_value[::, section] ^= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_sb_and_src(mask, sbs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_vrs_and_gl(diri: DIRI,
                                  vrs: Sequence[np.ndarray],
                                  gl: np.ndarray,
                                  sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GL
        expected_value[::, section] ^= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_sb_and_src(mask, vrs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_sbs_and_ggl(diri: DIRI,
                                   sbs: Sequence[int],
                                   ggl: np.ndarray,
                                   sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GGL[::, section // 4]
        expected_value[::, section] ^= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_sb_and_src(mask, sbs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_vrs_and_ggl(diri: DIRI,
                                   vrs: Sequence[np.ndarray],
                                   ggl: np.ndarray,
                                   sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GGL[::, section // 4]
        expected_value[::, section] ^= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_sb_and_src(mask, vrs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_sbs_and_rsp16(diri: DIRI,
                                     sbs: Sequence[int],
                                     rsp16: np.ndarray,
                                     sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] &= value
        expected_value[::, section] ^= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_sb_and_src(mask, sbs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_xor_equals_vrs_and_rsp16(diri: DIRI,
                                     vrs: Sequence[np.ndarray],
                                     rsp16: np.ndarray,
                                     sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] &= value
        expected_value[::, section] ^= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_xor_equals_sb_and_src(mask, vrs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# =================
# rl_from_sb_or_src
# =================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_or_rl(diri: DIRI,
                           sbs: Sequence[int],
                           rl: np.ndarray,
                           sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] |= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_or_src(mask, sbs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_or_rl(diri: DIRI,
                           vrs: Sequence[np.ndarray],
                           rl: np.ndarray,
                           sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] |= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_or_src(mask, vrs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_or_gl(diri: DIRI,
                           sbs: Sequence[int],
                           gl: np.ndarray,
                           sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] |= diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_or_src(mask, sbs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_or_gl(diri: DIRI,
                           vrs: Sequence[np.ndarray],
                           gl: np.ndarray,
                           sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] |= diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_or_src(mask, vrs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_or_ggl(diri: DIRI,
                            sbs: Sequence[int],
                            ggl: np.ndarray,
                            sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] |= diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_or_src(mask, sbs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_or_ggl(diri: DIRI,
                            vrs: Sequence[np.ndarray],
                            ggl: np.ndarray,
                            sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] |= diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_or_src(mask, vrs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_or_rsp16(diri: DIRI,
                              sbs: Sequence[int],
                              rsp16: np.ndarray,
                              sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] |= value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_or_src(mask, sbs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_or_rsp16(diri: DIRI,
                              vrs: Sequence[np.ndarray],
                              rsp16: np.ndarray,
                              sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] |= value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_or_src(mask, vrs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# ==================
# rl_from_sb_xor_src
# ==================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_xor_rl(diri: DIRI,
                            sbs: Sequence[int],
                            rl: np.ndarray,
                            sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] ^= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_src(mask, sbs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_xor_rl(diri: DIRI,
                            vrs: Sequence[np.ndarray],
                            rl: np.ndarray,
                            sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] ^= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_src(mask, vrs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_xor_gl(diri: DIRI,
                            sbs: Sequence[int],
                            gl: np.ndarray,
                            sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] ^= diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_src(mask, sbs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_xor_gl(diri: DIRI,
                            vrs: Sequence[np.ndarray],
                            gl: np.ndarray,
                            sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] ^= diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_src(mask, vrs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_xor_ggl(diri: DIRI,
                             sbs: Sequence[int],
                             ggl: np.ndarray,
                             sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] ^= diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_src(mask, sbs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_xor_ggl(diri: DIRI,
                             vrs: Sequence[np.ndarray],
                             ggl: np.ndarray,
                             sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] ^= diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_src(mask, vrs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_xor_rsp16(diri: DIRI,
                               sbs: Sequence[int],
                               rsp16: np.ndarray,
                               sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] ^= value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_src(mask, sbs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_xor_rsp16(diri: DIRI,
                               vrs: Sequence[np.ndarray],
                               rsp16: np.ndarray,
                               sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] ^= value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_src(mask, vrs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# ======================
# rl_from_sb_xor_inv_src
# ======================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_xor_inv_rl(diri: DIRI,
                                sbs: Sequence[int],
                                rl: np.ndarray,
                                sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] ^= ~diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_inv_src(mask, sbs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_xor_inv_rl(diri: DIRI,
                                vrs: Sequence[np.ndarray],
                                rl: np.ndarray,
                                sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] ^= ~diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_inv_src(mask, vrs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_xor_inv_gl(diri: DIRI,
                                sbs: Sequence[int],
                                gl: np.ndarray,
                                sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] ^= ~diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_inv_src(mask, sbs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_xor_inv_gl(diri: DIRI,
                                vrs: Sequence[np.ndarray],
                                gl: np.ndarray,
                                sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] ^= ~diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_inv_src(mask, vrs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_xor_inv_ggl(diri: DIRI,
                                 sbs: Sequence[int],
                                 ggl: np.ndarray,
                                 sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] ^= ~diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_inv_src(mask, sbs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_xor_inv_ggl(diri: DIRI,
                                 vrs: Sequence[np.ndarray],
                                 ggl: np.ndarray,
                                 sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] ^= ~diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_inv_src(mask, vrs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_xor_inv_rsp16(diri: DIRI,
                                   sbs: Sequence[int],
                                   rsp16: np.ndarray,
                                   sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] ^= ~value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_inv_src(mask, sbs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_xor_inv_rsp16(diri: DIRI,
                                   vrs: Sequence[np.ndarray],
                                   rsp16: np.ndarray,
                                   sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] ^= ~value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_xor_inv_src(mask, vrs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# ======================
# rl_from_inv_sb_and_src
# ======================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_sbs_and_rl(diri: DIRI,
                                sbs: Sequence[int],
                                rl: np.ndarray,
                                sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~diri.hb[sb, ::, section]
            else:
                expected_value[::, section] |= ~diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_src(mask, sbs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_vrs_and_rl(diri: DIRI,
                                vrs: Sequence[np.ndarray],
                                rl: np.ndarray,
                                sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~vr[::, section]
            else:
                expected_value[::, section] |= ~vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_src(mask, vrs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_sbs_and_gl(diri: DIRI,
                                sbs: Sequence[int],
                                gl: np.ndarray,
                                sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~diri.hb[sb, ::, section]
            else:
                expected_value[::, section] |= ~diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_src(mask, sbs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_vrs_and_gl(diri: DIRI,
                                vrs: Sequence[np.ndarray],
                                gl: np.ndarray,
                                sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~vr[::, section]
            else:
                expected_value[::, section] |= ~vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_src(mask, vrs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_sbs_and_ggl(diri: DIRI,
                                 sbs: Sequence[int],
                                 ggl: np.ndarray,
                                 sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~diri.hb[sb, ::, section]
            else:
                expected_value[::, section] |= ~diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_src(mask, sbs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_vrs_and_ggl(diri: DIRI,
                                 vrs: Sequence[np.ndarray],
                                 ggl: np.ndarray,
                                 sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~vr[::, section]
            else:
                expected_value[::, section] |= ~vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_src(mask, vrs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_sbs_and_rsp16(diri: DIRI,
                                   sbs: Sequence[int],
                                   rsp16: np.ndarray,
                                   sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~diri.hb[sb, ::, section]
            else:
                expected_value[::, section] |= ~diri.hb[sb, ::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] &= value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_src(mask, sbs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_vrs_and_rsp16(diri: DIRI,
                                   vrs: Sequence[np.ndarray],
                                   rsp16: np.ndarray,
                                   sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~vr[::, section]
            else:
                expected_value[::, section] |= ~vr[::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] &= value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_src(mask, vrs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# ======================
# rl_from_sb_xor_inv_src
# ======================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_sbs_and_inv_rl(diri: DIRI,
                                    sbs: Sequence[int],
                                    rl: np.ndarray,
                                    sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~diri.hb[sb, ::, section]
            else:
                expected_value[::, section] |= ~diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= ~diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_inv_src(mask, sbs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_vrs_and_inv_rl(diri: DIRI,
                                    vrs: Sequence[np.ndarray],
                                    rl: np.ndarray,
                                    sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~vr[::, section]
            else:
                expected_value[::, section] |= ~vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= ~diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_inv_src(mask, vrs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_sbs_and_inv_gl(diri: DIRI,
                                    sbs: Sequence[int],
                                    gl: np.ndarray,
                                    sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~diri.hb[sb, ::, section]
            else:
                expected_value[::, section] |= ~diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= ~diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_inv_src(mask, sbs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_vrs_and_inv_gl(diri: DIRI,
                                    vrs: Sequence[np.ndarray],
                                    gl: np.ndarray,
                                    sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~vr[::, section]
            else:
                expected_value[::, section] |= ~vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= ~diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_inv_src(mask, vrs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_sbs_and_inv_ggl(diri: DIRI,
                                     sbs: Sequence[int],
                                     ggl: np.ndarray,
                                     sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~diri.hb[sb, ::, section]
            else:
                expected_value[::, section] |= ~diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= ~diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_inv_src(mask, sbs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_vrs_and_inv_ggl(diri: DIRI,
                                     vrs: Sequence[np.ndarray],
                                     ggl: np.ndarray,
                                     sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~vr[::, section]
            else:
                expected_value[::, section] |= ~vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= ~diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_inv_src(mask, vrs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_sbs_and_inv_rsp16(diri: DIRI,
                                       sbs: Sequence[int],
                                       rsp16: np.ndarray,
                                       sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~diri.hb[sb, ::, section]
            else:
                expected_value[::, section] |= ~diri.hb[sb, ::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] &= ~value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_inv_src(mask, sbs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_inv_vrs_and_inv_rsp16(diri: DIRI,
                                       vrs: Sequence[np.ndarray],
                                       rsp16: np.ndarray,
                                       sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = ~vr[::, section]
            else:
                expected_value[::, section] |= ~vr[::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] &= ~value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_inv_sb_and_inv_src(mask, vrs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# ======================
# rl_from_sb_and_inv_src
# ======================


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_and_inv_rl(diri: DIRI,
                                sbs: Sequence[int],
                                rl: np.ndarray,
                                sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= ~diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_inv_src(mask, sbs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_and_inv_rl(diri: DIRI,
                                vrs: Sequence[np.ndarray],
                                rl: np.ndarray,
                                sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= ~diri.RL()[::, section]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_inv_src(mask, vrs, diri.RL())
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_and_inv_gl(diri: DIRI,
                                sbs: Sequence[int],
                                gl: np.ndarray,
                                sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= ~diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_inv_src(mask, sbs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_and_inv_gl(diri: DIRI,
                                vrs: Sequence[np.ndarray],
                                gl: np.ndarray,
                                sections: int) -> None:
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= ~diri.GL
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_inv_src(mask, vrs, diri.GL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_and_inv_ggl(diri: DIRI,
                                 sbs: Sequence[int],
                                 ggl: np.ndarray,
                                 sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        expected_value[::, section] &= ~diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_inv_src(mask, sbs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_and_inv_ggl(diri: DIRI,
                                 vrs: Sequence[np.ndarray],
                                 ggl: np.ndarray,
                                 sections: int) -> None:
    diri.GGL[::, ::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        expected_value[::, section] &= ~diri.GGL[::, section // 4]
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_inv_src(mask, vrs, diri.GGL)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    sbs=sbs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_sbs_and_inv_rsp16(diri: DIRI,
                                   sbs: Sequence[int],
                                   rsp16: np.ndarray,
                                   sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_vr()
    for index, sb in enumerate(sbs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = diri.hb[sb, ::, section]
            else:
                expected_value[::, section] &= diri.hb[sb, ::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] &= ~value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_inv_src(mask, sbs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


@diri_property_test(repeatably_randomize_half_bank=True,
                    vrs=vrs_strategy,
                    rsp16=rsp16_strategy(),
                    sections=Mask_strategy())
def test_rl_from_vrs_and_inv_rsp16(diri: DIRI,
                                   vrs: Sequence[np.ndarray],
                                   rsp16: np.ndarray,
                                   sections: int) -> None:
    diri.RSP16[::, ::] = convert_to_bool(rsp16)
    mask = Mask(f"0x{sections:04X}")
    vrs = list(map(convert_to_bool, vrs))
    expected_value = diri.build_vr()
    for index, vr in enumerate(vrs):
        for section in list(mask):
            if index == 0:
                expected_value[::, section] = vr[::, section]
            else:
                expected_value[::, section] &= vr[::, section]
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            value = diri.RSP16[index, section]
            expected_value[start:stop, section] &= ~value
    for section in list(~mask):
        expected_value[::, section] = diri.RL()[::, section]
    diri.rl_from_sb_and_inv_src(mask, vrs, diri.RSP16)
    assert np.array_equal(expected_value, diri.RL())


# =============
# rsp16_from_rl
# =============


@diri_property_test(rl=vr_strategy(),
                    sections=Mask_strategy())
def test_rsp16_from_rl(diri: DIRI,
                       rl: np.ndarray,
                       sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_rsp16()
    for section in list(mask):
        for index in range(NUM_PLATS_PER_APUC // 16):
            start = index * 16
            stop = start + 16
            expected_value[index, section] = diri.RL()[start:stop, section].any()
    diri.rsp16_from_rl(mask)
    assert np.array_equal(diri.RSP16, expected_value)


# ==========
# gl_from_rl
# ==========


@diri_property_test(rl=vr_strategy(),
                    gl=gl_strategy(),
                    sections=Mask_strategy())
def test_gl_from_rl(diri: DIRI,
                    rl: np.ndarray,
                    gl: np.ndarray,
                    sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GL[::] = np.squeeze(convert_to_bool(gl, nsections=1))
    mask = Mask(f"0x{sections:04X}")
    expected_value = diri.build_gl()
    expected_value[::] = diri.RL()[::, list(mask)].all(axis=1)
    diri.gl_from_rl(mask)
    assert np.array_equal(diri.GL, expected_value)


# ===========
# ggl_from_rl
# ===========


@diri_property_test(rl=vr_strategy(),
                    ggl=ggl_strategy(),
                    sections=Mask_strategy())
def test_ggl_from_rl(diri: DIRI,
                     rl: np.ndarray,
                     ggl: np.ndarray,
                     sections: int) -> None:
    diri.RL()[::, ::] = convert_to_bool(rl)
    diri.GGL[::] = convert_to_bool(ggl, nsections=4)
    mask = Mask(f"0x{sections:04X}")
    expected_value = np.ones(GGL_SHAPE, dtype=bool)
    for section in list(mask):
        expected_value[::, section // 4] &= diri.RL()[::, section]
    diri.ggl_from_rl(mask)
    assert np.array_equal(diri.GGL, expected_value)


# ===========
# l1_from_ggl
# ===========


@diri_property_test(ggl=ggl_strategy(),
                    l1=ggl_strategy(),
                    l1_addr=l1_addr_strategy)
def test_l1_from_ggl(diri: DIRI,
                     ggl: np.ndarray,
                     l1: np.ndarray,
                     l1_addr: int) -> None:
    diri.GGL[::] = convert_to_bool(ggl, nsections=4)
    diri.L1[l1_addr, ::, ::] = convert_to_bool(l1, nsections=4)
    expected_value = deepcopy(diri.GGL)
    diri.l1_from_ggl(l1_addr)
    assert np.array_equal(diri.L1[l1_addr], expected_value)


# ===========
# lgl_from_l1
# ===========


@diri_property_test(lgl=lgl_strategy(),
                    l1=ggl_strategy(),
                    l1_addr=l1_addr_strategy)
def test_lgl_from_l1(diri: DIRI,
                     lgl: np.ndarray,
                     l1: np.ndarray,
                     l1_addr: int) -> None:
    diri.LGL[::, None] = convert_to_bool(lgl, nsections=1)
    diri.L1[l1_addr, ::, ::] = convert_to_bool(l1, nsections=4)
    bank = (l1_addr >> 11) & 0b11
    group = (l1_addr >> 9) & 0b11
    row = l1_addr & 0b111111111
    plats = plats_for_bank(bank)
    expected_value = deepcopy(diri.L1[row, plats, group])
    diri.lgl_from_l1(l1_addr)
    assert np.array_equal(diri.LGL, expected_value)


# ===========
# l2_from_lgl
# ===========


@diri_property_test(lgl=lgl_strategy(),
                    l2=lgl_strategy(),
                    l2_addr=l2_addr_strategy)
def test_l2_from_lgl(diri: DIRI,
                     lgl: np.ndarray,
                     l2: np.ndarray,
                     l2_addr: int) -> None:
    diri.LGL[::, None] = convert_to_bool(lgl, nsections=1)
    diri.L2[l2_addr] = convert_to_bool(l2, nsections=1)[::, 0]
    expected_value = deepcopy(diri.LGL)
    diri.l2_from_lgl(l2_addr)
    assert np.array_equal(diri.L2[l2_addr], expected_value)


# ===========
# lgl_from_l2
# ===========


@diri_property_test(lgl=lgl_strategy(),
                    l2=lgl_strategy(),
                    l2_addr=l2_addr_strategy)
def test_lgl_from_l2(diri: DIRI,
                     lgl: np.ndarray,
                     l2: np.ndarray,
                     l2_addr: int) -> None:
    diri.LGL[::, None] = convert_to_bool(lgl, nsections=1)
    diri.L2[l2_addr] = convert_to_bool(l2, nsections=1)[::, 0]
    expected_value = deepcopy(diri.L2[l2_addr])
    diri.lgl_from_l2(l2_addr)
    assert np.array_equal(diri.LGL, expected_value)


# ===========
# l1_from_lgl
# ===========


@diri_property_test(lgl=lgl_strategy(),
                    l1=ggl_strategy(),
                    l1_addr=l1_addr_strategy)
def test_l1_from_lgl(diri: DIRI,
                     lgl: np.ndarray,
                     l1: np.ndarray,
                     l1_addr: int) -> None:
    diri.LGL[::, None] = convert_to_bool(lgl, nsections=1)
    diri.L1[l1_addr, ::, ::] = convert_to_bool(l1, nsections=4)
    bank = (l1_addr >> 11) & 0b11
    group = (l1_addr >> 9) & 0b11
    row = l1_addr & 0b111111111
    plats = plats_for_bank(bank)
    expected_value = deepcopy(diri.LGL)
    diri.l1_from_lgl(l1_addr)
    assert np.array_equal(diri.L1[row, plats, group], expected_value)


# ===========
# ggl_from_l1
# ===========


@diri_property_test(ggl=ggl_strategy(),
                    l1=ggl_strategy(),
                    l1_addr=l1_addr_strategy)
def test_ggl_from_l1(diri: DIRI,
                     ggl: np.ndarray,
                     l1: np.ndarray,
                     l1_addr: int) -> None:
    diri.GGL[::] = convert_to_bool(ggl, nsections=4)
    diri.L1[l1_addr, ::, ::] = convert_to_bool(l1, nsections=4)
    expected_value = deepcopy(diri.L1[l1_addr])
    diri.ggl_from_l1(l1_addr)
    assert np.array_equal(diri.GGL, expected_value)


# ==================
# ggl_from_rl_and_l1
# ==================


# FIXME: Determine the proper behavior of the FUT and update the test accordingly
# @diri_property_test(ggl=ggl_strategy(),
#                     rl=vr_strategy(),
#                     l1=ggl_strategy(),
#                     l1_addr=l1_addr_strategy,
#                     sections=Mask_strategy())
# def test_ggl_from_rl_and_l1(diri: DIRI,
#                             ggl: np.ndarray,
#                             rl: np.ndarray,
#                             l1: np.ndarray,
#                             l1_addr: int,
#                             sections: int) -> None:
#     diri.RL()[::, ::] = convert_to_bool(rl)
#     diri.GGL[::] = convert_to_bool(ggl, nsections=4)
#     diri.L1[l1_addr, ::, ::] = convert_to_bool(l1, nsections=4)
#     mask = Mask(sections)
#     if sections == 0x0000:
#         expected_value = np.zeros(GGL_SHAPE, dtype=bool)
#     else:
#         expected_value = np.ones(GGL_SHAPE, dtype=bool)
#     for section in list(mask):
#         expected_value[::, section // 4] &= diri.RL()[::, section]
#     expected_value[::, ::] &= diri.L1[l1_addr]
#     diri.ggl_from_rl_and_l1(mask, l1_addr)
#     assert np.array_equal(diri.GGL, expected_value)


# =================
# getters / setters
# =================


@diri_test(repeatably_randomize_half_bank=True)
def test_diri_getter(diri: DIRI) -> None:
    # A single parameter is the SB number of the first half-bank, the VR will
    # have shape 2048 x 16
    assert diri[0].shape == (NPLATS, NSECTIONS)
    assert np.array_equal(diri[0],
                          diri.hb[0, :NPLATS, :NSECTIONS])
    assert np.array_equal(diri[0, 0, ::, ::],
                          diri.hb[0, :NPLATS, :NSECTIONS])

    # Two parameters represent the SB and plat indices
    assert diri[0, :32].shape == (32, NSECTIONS)
    assert np.array_equal(diri[0, :32],
                          diri.hb[0, :32, :NSECTIONS])
    assert np.array_equal(diri[0, 0, :32, ::],
                          diri.hb[0, :32, :NSECTIONS])

    # Three parameters represent the SB, plat, and section indices
    assert diri[0, :32, :8].shape == (32, 8)
    assert np.array_equal(diri[0, :32, :8],
                          diri.hb[0, :32, :8])
    assert np.array_equal(diri[0, 0, :32, :8],
                          diri.hb[0, :32, :8])

    # Four parameters represent the half-bank, SB, plat, and section indices
    assert diri[0, 0, :32, :8].shape == (32, 8)
    assert np.array_equal(diri[0, 0, :32, :8],
                          diri.hb[0, :32, :8])

    # If you want to get the whole 2048 x 16 VR of half-bank 5, SB 12:
    assert diri[5, 12, ::, ::].shape == (NPLATS, NSECTIONS)
    assert np.array_equal(diri[5, 12, ::, ::],
                          diri.hb[12, (5 * NPLATS):(6 * NPLATS), :NSECTIONS])

    # If you want to get the 4096 x 16 VR of half-banks 3 and 7, SB 6:
    assert diri[[3, 7], 6, ::, ::].shape == (2 * NPLATS, NSECTIONS)
    assert np.array_equal(diri[[3, 7], 6, ::, ::],
                          diri.hb[6,
                                  list(chain(range(3 * NPLATS, 4 * NPLATS),
                                             range(7 * NPLATS, 8 * NPLATS))),
                                  :NSECTIONS])

    # If you want all 32k plats of SB 3:
    assert diri[::, 3, ::, ::].shape == (NUM_PLATS_PER_APUC, NSECTIONS)
    assert np.array_equal(diri[::, 3, ::, ::],
                          diri.hb[3, ::, ::])

    # If you want SBs 1 and 2 of half-banks 4 and 8 (more than one SB increases
    # the dimensionality of the return value):
    assert diri[[4, 8], [1, 2], ::, ::].shape == (2, 2 * NPLATS, NSECTIONS)
    assert np.array_equal(
        diri[[4, 8], [1, 2], ::, ::],
        np.array([
            diri.hb[1,
                    list(chain(range(4 * NPLATS, 5 * NPLATS),
                               range(8 * NPLATS, 9 * NPLATS))),
                    :NSECTIONS],
            diri.hb[2,
                    list(chain(range(4 * NPLATS, 5 * NPLATS),
                               range(8 * NPLATS, 9 * NPLATS))),
                    :NSECTIONS],
        ]))

    # Etc.


@diri_property_test(repeatably_randomize_half_bank=True,
                    data=st.data())
def test_diri_setter(diri: DIRI, data: object) -> None:
    # Assign a 2048 x 16 VR to a specific SB of half-bank 0:
    # A single parameter specifies the SB number of half-bank 0.
    old_vr = deepcopy(diri[::, 0, ::, ::])
    hb_0_vr_0 = convert_to_bool(data.draw(vr_strategy(num_plats=NPLATS)))
    diri[0] = hb_0_vr_0
    new_vr = diri[::, 0, ::, ::]
    assert np.array_equal(hb_0_vr_0, new_vr[:NPLATS])
    assert np.array_equal(old_vr[NPLATS:], new_vr[NPLATS:])

    # Assign a 2048 x 16 VR to a specific SB of half-bank 3:
    # Four parameters specify the half-bank, SB, plat, and section indices,
    # respectfully.
    old_vr = deepcopy(diri[::, 0, ::, ::])
    hb_3_vr_0 = convert_to_bool(data.draw(vr_strategy(num_plats=NPLATS)))
    diri[3, 0, ::, ::] = hb_3_vr_0
    new_vr = diri[::, 0, ::, ::]
    assert np.array_equal(old_vr[:(3 * NPLATS)], new_vr[:(3 * NPLATS)])
    assert np.array_equal(hb_3_vr_0, new_vr[(3 * NPLATS):(4 * NPLATS)])
    assert np.array_equal(old_vr[(4 * NPLATS):], new_vr[(4 * NPLATS):])

    # Assign a full 32768 x 16 VR to a specific SB of all half-banks:
    vr_3 = convert_to_bool(data.draw(vr_strategy()))
    diri[::, 3, ::, ::] = vr_3
    assert np.array_equal(vr_3, diri[::, 3, ::, ::])

    # Assign a glass pattern to part of SB 5 of half-bank 0:
    old_vr = deepcopy(diri[::, 5, ::, ::])
    diri[5, 3:(3+7), 5:(5+4)] = [  # Also works with newline-delimited strings
        "0110011",
        "1110111",
        "1010101",
        "0001101",
    ]
    assert np.array_equal(diri[5, :3, :5], old_vr[:3, :5])
    assert np.array_equal(diri[5, :3, (5+4):], old_vr[:3, (5+4):])
    assert np.array_equal(diri[5, 3:(3+7), 5:(5+4)], [
        [False, True, True, False],
        [True, True, False, False],
        [True, True, True, False],
        [False, False, False, True],
        [False, True, True, True],
        [True, True, False, False],
        [True, True, True, True],
    ])
    assert np.array_equal(diri[5, (3+7):, :5], old_vr[(3+7):, :5])
    assert np.array_equal(diri[5, (3+7):, (5+4):], old_vr[(3+7):, (5+4):])

    # Etc.


@diri_property_test(data=vr_strategy(),
                    row_number=sb_strategy,
                    vmr_addr=vmr_strategy)
def test_diri_vmr(diri: DIRI,
                  data: np.ndarray,
                  row_number: int,
                  vmr_addr: int) -> None:
    vr = convert_to_bool(data)
    diri.hb[row_number, ::, ::] = vr
    vm = BLEIRVirtualMachine(diri=diri,
                             interpret=True,
                             generate_code=False)
    vm.compile(store_16(vmr_addr, row_number))
    vmr = diri.vmr[vmr_addr]
    assert np.array_equal(vmr, vr)


@belex_apl
def read_src_into_rsp32k(Belex, src: VR):
    RL[::] <= src()
    RSP16[::] <= RL()
    RSP256() <= RSP16()
    RSP2K() <= RSP256()
    RSP32K() <= RSP2K()
    NOOP()
    NOOP()
    RSP_END()


def test_rsp_fifo() -> None:
    apuc_rsp_fifo = ApucRspFifo()
    diri = DIRI.push_context(apuc_rsp_fifo=apuc_rsp_fifo)
    try:
        src_vp = 0
        for half_bank_id in range(NUM_HALF_BANKS_PER_APUC):
            lower_plat = half_bank_id * NUM_PLATS_PER_HALF_BANK
            upper_plat = lower_plat + NUM_PLATS_PER_HALF_BANK
            diri.hb[src_vp, lower_plat:upper_plat, ::] = \
                convert_to_bool(half_bank_id)

        read_src_into_rsp32k(src_vp)

        for apc_id in range(2):
            apuc_rsp_fifo.rsp_rd(apc_id)
            for bank_id in range(4):
                rsp2k_val = apuc_rsp_fifo.rd_rsp2k_reg(bank_id)
                for half_bank_id in range(2):
                    half_bank_val = \
                        (rsp2k_val >> (16 * half_bank_id)) & 0xFFFF
                    assert half_bank_val == \
                        apc_id * 8 + bank_id + half_bank_id * 4
            rsp32k_val = apuc_rsp_fifo.rd_rsp32k_reg()
            assert rsp32k_val == (0b11111110 + apc_id)

        for _ in range(FIFO_CAPACITY):
            read_src_into_rsp32k(src_vp)
        # Should raise an exception when RSP fifo is full
        with pytest.raises(RuntimeError):
            read_src_into_rsp32k(src_vp)
    finally:
        assert diri is DIRI.pop_context()
