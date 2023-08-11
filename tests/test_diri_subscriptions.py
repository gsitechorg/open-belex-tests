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

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional, Sequence, Set

import numpy as np

import hypothesis
import hypothesis.strategies as st
from hypothesis import given

from open_belex.common.constants import (MAX_EWE_VALUE, MAX_L2_VALUE,
                                         MAX_RE_VALUE, NGGL_ROWS, NSB,
                                         NSECTIONS, NUM_EWE_REGS, NUM_L1_REGS,
                                         NUM_L2_REGS, NUM_PLATS_PER_APUC,
                                         NUM_PLATS_PER_HALF_BANK, NUM_RE_REGS,
                                         NUM_RN_REGS, NUM_SM_REGS)
from open_belex.common.rsp_fifo import ApucRspFifo
from open_belex.common.seu_layer import SEULayer
from open_belex.diri.half_bank import DIRI, VALID_L1_ADDRESSES
from open_belex.kernel_libs.memory import (copy_l1_to_l2_byte, l2_end,
                                           store_16_t0)
from open_belex.literal import (EWE, EWE_REG_0, EWE_REG_1, EWE_REG_2,
                                EWE_REG_3, GL, L1, L1_ADDR_REG_0,
                                L1_ADDR_REG_1, L1_ADDR_REG_2, L1_ADDR_REG_3,
                                L2, L2_ADDR_REG_0, NOOP, RE, RE_REG_0,
                                RE_REG_1, RE_REG_2, RE_REG_3, RL, RN_REG_0,
                                RN_REG_1, RN_REG_2, RN_REG_3, RN_REG_4,
                                RN_REG_5, RN_REG_6, RN_REG_7, RN_REG_8,
                                RN_REG_9, RN_REG_10, RN_REG_11, RN_REG_12,
                                RN_REG_13, RN_REG_14, RN_REG_15, RSP2K, RSP16,
                                RSP32K, RSP256, RSP_END, RWINH_RST, RWINH_SET,
                                SM_REG_0, SM_REG_1, SM_REG_2, SM_REG_3,
                                SM_REG_4, SM_REG_5, SM_REG_6, SM_REG_7,
                                SM_REG_8, SM_REG_9, SM_REG_10, SM_REG_11,
                                SM_REG_12, SM_REG_13, SM_REG_14, SM_REG_15, VR,
                                Mask, apl_set_ewe_reg, apl_set_l1_reg,
                                apl_set_l2_reg, apl_set_re_reg, apl_set_rn_reg,
                                apl_set_sm_reg, belex_apl)
from open_belex.utils.memory_utils import NUM_VM_REGS

from open_belex_libs.common import cpy_imm_16
from open_belex_libs.memory import (belex_gal_vm_reg_to_set_ext,
                                    store_16_parity_mask)

GSI_L1_NUM_GRPS = 4
BELEX_L2_CTL_ROW_ADDR_BIT_IDX_BITS = 4


def coerce_ilist(length: int) -> Callable[[Optional[Sequence[int]]],
                                          Sequence[int]]:
    def coerce_fn(indices: Optional[Sequence[int]]) -> Sequence[int]:
        if indices is None:
            indices = list(range(length))
        return indices
    return coerce_fn


vr_plats = coerce_ilist(NUM_PLATS_PER_APUC)
vr_sections = coerce_ilist(NSECTIONS)

rl_plats = coerce_ilist(NUM_PLATS_PER_APUC)
rl_sections = coerce_ilist(NSECTIONS)

gl_plats = coerce_ilist(NUM_PLATS_PER_APUC)

ggl_plats = coerce_ilist(NUM_PLATS_PER_APUC)
ggl_groups = coerce_ilist(NGGL_ROWS)

rsp16_plats = coerce_ilist(NUM_PLATS_PER_APUC // 16)
rsp16_sections = coerce_ilist(NSECTIONS)

rsp256_plats = coerce_ilist(NUM_PLATS_PER_APUC // 256)
rsp256_sections = coerce_ilist(NSECTIONS)

rsp2k_plats = coerce_ilist(NUM_PLATS_PER_APUC // 2048)
rsp2k_sections = coerce_ilist(NSECTIONS)

rsp32k_plats = coerce_ilist(NUM_PLATS_PER_APUC // 32768)
rsp32k_sections = coerce_ilist(NSECTIONS)

l1_plats = coerce_ilist(NUM_PLATS_PER_APUC)
l1_groups = coerce_ilist(NGGL_ROWS)

l2_plats = coerce_ilist(NUM_PLATS_PER_HALF_BANK * 4)

lgl_plats = coerce_ilist(NUM_PLATS_PER_HALF_BANK * 4)


@dataclass
class EventRecorder:
    events: Deque[Any] = field(default_factory=deque)

    def capture_events(self: "EventRecorder",
                       seu_layer: SEULayer,
                       apuc_rsp_fifo: ApucRspFifo,
                       diri: DIRI) -> None:
        seu_layer.subscribe(self.events.append)
        apuc_rsp_fifo.subscribe(self.events.append)
        diri.subscribe(self.events.append)

    def replay_events(self: "EventRecorder",
                      seu_layer: SEULayer,
                      apuc_rsp_fifo: ApucRspFifo,
                      diri: DIRI) -> None:
        for event in self.events:
            match event:
                # -----------------
                # seu_layer events:
                # -----------------
                case ("seu::sm_reg", reg_id, value):
                    seu_layer.sm_regs[reg_id] = value
                case ("seu::rn_reg", reg_id, value):
                    seu_layer.rn_regs[reg_id] = value
                case ("seu::re_reg", reg_id, value):
                    seu_layer.re_regs[reg_id] = value
                case ("seu::ewe_reg", reg_id, value):
                    seu_layer.ewe_regs[reg_id] = value
                case ("seu::l1_addr_reg", reg_id, value):
                    seu_layer.l1_regs[reg_id] = value
                case ("seu::l2_addr_reg", reg_id, value):
                    seu_layer.l2_regs[reg_id] = value
                # ---------------------
                # apuc_rsp_fifo events:
                # ---------------------
                case ("fifo::enqueue", apc_id, rsp_fifo_msg, length):
                    apc_rsp_fifo = apuc_rsp_fifo.queues[apc_id]
                    apuc_rsp_fifo.active = apc_rsp_fifo
                    apc_rsp_fifo.enqueue(rsp_fifo_msg)
                    assert length == apc_rsp_fifo.length - 1
                case ("fifo::dequeue", apc_id, length):
                    apc_rsp_fifo = apuc_rsp_fifo.queues[apc_id]
                    apuc_rsp_fifo.active = apc_rsp_fifo
                    apc_rsp_fifo.rsp_rd()
                    assert length == apc_rsp_fifo.length - 1
                # ------------
                # diri events:
                # ------------
                case ("diri::rw_inh_filter", plats, sections, value):
                    plats, sections = vr_plats(plats), vr_sections(sections)
                    if np.ndim(value) == 1:
                        value = value[None].T
                    diri.rwinh_filter[np.ix_(plats, sections)] = value
                case ("diri::vr", row_number, plats, sections, value):
                    plats, sections = vr_plats(plats), vr_sections(sections)
                    if np.ndim(value) == 1:
                        value = value[None].T
                    diri.hb.vrs[row_number][np.ix_(plats, sections)] = value
                case ("diri::rl", plats, sections, value):
                    plats, sections = rl_plats(plats), rl_sections(sections)
                    if np.ndim(value) == 1:
                        value = value[None].T
                    diri.hb.rl[np.ix_(plats, sections)] = value
                case ("diri::gl", plats, value):
                    plats = gl_plats(plats)
                    diri.GL[plats] = value
                case ("diri::ggl", plats, groups, value):
                    plats, groups = ggl_plats(plats), ggl_groups(groups)
                    if np.ndim(value) == 1:
                        value = value[None].T
                    diri.GGL[np.ix_(plats, groups)] = value
                case ("diri::rsp16", plats, sections, value):
                    plats, sections = \
                        rsp16_plats(plats), rsp16_sections(sections)
                    if np.ndim(value) == 1:
                        value = value[None].T
                    diri.RSP16[np.ix_(plats, sections)] = value
                case ("diri::rsp256", plats, sections, value):
                    plats, sections = \
                        rsp256_plats(plats), rsp256_sections(sections)
                    if np.ndim(value) == 1:
                        value = value[None].T
                    diri.RSP256[np.ix_(plats, sections)] = value
                case ("diri::rsp2k", plats, sections, value):
                    plats, sections = \
                        rsp2k_plats(plats), rsp2k_sections(sections)
                    if np.ndim(value) == 1:
                        value = value[None].T
                    diri.RSP2K[np.ix_(plats, sections)] = value
                case ("diri::rsp32k", plats, sections, value):
                    plats, sections = \
                        rsp32k_plats(plats), rsp32k_sections(sections)
                    if np.ndim(value) == 1:
                        value = value[None].T
                    diri.RSP32K[np.ix_(plats, sections)] = value
                case ("diri::l1", l1_addr, plats, sections, value):
                    plats, groups = l1_plats(plats), l1_groups(groups)
                    if np.ndim(value) == 1:
                        value = value[None].T
                    diri.L1[l1_addr][np.ix_(plats, groups)] = value
                case ("diri::l2", l2_addr, plats, value):
                    plats = l2_plats(plats)
                    diri.L2[l2_addr][plats] = value
                case ("diri::lgl", plats, value):
                    plats = lgl_plats(plats)
                    diri.LGL[plats] = value
                case unmatched_event:
                    raise RuntimeError(
                        f"Cannot replay unsupported event: {unmatched_event}")


def diri_subscription_test(test_fn: Optional[Callable] = None,
                           max_examples: int = 5,
                           **kwargs) -> Callable:

    def decorator(test_fn: Callable) -> Callable:

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
        @given(data=st.data(), **kwargs)
        def wrapper(data, *args, **kwargs) -> None:
            ctx_seu_layer = SEULayer.push_context()
            ctx_apuc_rsp_fifo = ApucRspFifo()
            ctx_diri = DIRI.push_context(apuc_rsp_fifo=ctx_apuc_rsp_fifo)

            event_recorder = EventRecorder()
            event_recorder.capture_events(ctx_seu_layer, ctx_apuc_rsp_fifo, ctx_diri)

            test_fn(data, ctx_seu_layer, ctx_apuc_rsp_fifo, ctx_diri, *args, **kwargs)

            assert ctx_diri is DIRI.pop_context()
            assert ctx_seu_layer is SEULayer.pop_context()

            alt_seu_layer = SEULayer()
            alt_apuc_rsp_fifo = ApucRspFifo()
            alt_diri = DIRI(apuc_rsp_fifo=alt_apuc_rsp_fifo)

            event_recorder.replay_events(alt_seu_layer, alt_apuc_rsp_fifo, alt_diri)

            assert ctx_seu_layer == alt_seu_layer
            assert ctx_apuc_rsp_fifo == alt_apuc_rsp_fifo
            assert ctx_diri == alt_diri

        wrapper.__fn__ = test_fn
        wrapper.__name__ = test_fn.__name__
        return wrapper

    if test_fn is not None:
        return decorator(test_fn)

    return decorator


st_sm_reg_id = st.integers(min_value=0, max_value=NUM_SM_REGS - 1)
st_rn_reg_id = st.integers(min_value=0, max_value=NUM_RN_REGS - 1)
st_re_reg_id = st.integers(min_value=0, max_value=NUM_RE_REGS - 1)
st_ewe_reg_id = st.integers(min_value=0, max_value=NUM_EWE_REGS - 1)
st_l1_reg_id = st.integers(min_value=0, max_value=NUM_L1_REGS - 1)
st_l2_reg_id = st.integers(min_value=0, max_value=NUM_L2_REGS - 1)

SM_REG_BY_ID = [globals()[f"SM_REG_{sm_reg_id}"]
                for sm_reg_id in range(NUM_SM_REGS)]
RN_REG_BY_ID = [globals()[f"RN_REG_{rn_reg_id}"]
                for rn_reg_id in range(NUM_RN_REGS)]
RE_REG_BY_ID = [globals()[f"RE_REG_{re_reg_id}"]
                for re_reg_id in range(NUM_RE_REGS)]
EWE_REG_BY_ID = [globals()[f"EWE_REG_{ewe_reg_id}"]
                 for ewe_reg_id in range(NUM_EWE_REGS)]
L1_REG_BY_ID = [globals()[f"L1_ADDR_REG_{l1_reg_id}"]
                for l1_reg_id in range(NUM_L1_REGS)]
L2_REG_BY_ID = [globals()[f"L2_ADDR_REG_{l2_reg_id}"]
                for l2_reg_id in range(NUM_L2_REGS)]


@st.composite
def st_sm_reg(draw) -> Mask:
    sm_reg_id = draw(st_sm_reg_id)
    return SM_REG_BY_ID[sm_reg_id]


@st.composite
def st_rn_reg(draw) -> VR:
    rn_reg_id = draw(st_rn_reg_id)
    return RN_REG_BY_ID[rn_reg_id]


@st.composite
def st_re_reg(draw) -> RE:
    re_reg_id = draw(st_re_reg_id)
    return RE_REG_BY_ID[re_reg_id]


@st.composite
def st_ewe_reg(draw) -> EWE:
    ewe_reg_id = draw(st_ewe_reg_id)
    return EWE_REG_BY_ID[ewe_reg_id]


@st.composite
def st_l1_reg(draw, exclude: Optional[Set[int]] = None) -> L1:
    if exclude is not None:
        l1_reg_id = draw(
            st.sampled_from(
                list(set(range(NUM_L1_REGS)) - exclude)))
    else:
        l1_reg_id = draw(st_l1_reg_id)
    return L1_REG_BY_ID[l1_reg_id]


@st.composite
def st_l2_reg(draw) -> L2:
    l2_reg_id = draw(st_l2_reg_id)
    return L2_REG_BY_ID[l2_reg_id]


st_rn_val = st.integers(min_value=0, max_value=NSB - 1)
st_sm_val = st.integers(min_value=0x0000, max_value=0xFFFF)
st_re_val = st.integers(min_value=0x000000, max_value=MAX_RE_VALUE)
st_ewe_val = st.integers(min_value=0x000, max_value=MAX_EWE_VALUE)
st_l1_val = st.sampled_from(np.where(VALID_L1_ADDRESSES)[0])
st_l2_val = st.integers(min_value=0, max_value=MAX_L2_VALUE)

st_vmr_val = st.integers(min_value=0, max_value=NUM_VM_REGS - 1)

st_seed = st.integers(min_value=0, max_value=2**32 - 1)

@st.composite
def st_np_random(draw, seed: Optional[int] = None) -> np.random.RandomState:
    if seed is None:
        seed = draw(st_seed)
    return np.random.RandomState(seed)


@st.composite
def st_vr_data(draw) -> np.ndarray:
    random = draw(st_np_random())
    return random.choice(a=[False, True],
                         size=(NUM_PLATS_PER_APUC, NSECTIONS),
                         p=[.5, .5])


@st.composite
def st_l1_data(draw) -> np.ndarray:
    random = draw(st_np_random())
    return random.choice(a=[False, True],
                         size=(NUM_PLATS_PER_APUC, NGGL_ROWS),
                         p=[.5, .5])


@st.composite
def st_l2_data(draw) -> np.ndarray:
    random = draw(st_np_random())
    return random.choice(a=[False, True],
                         size=NUM_PLATS_PER_HALF_BANK * 4,
                         p=[.5, .5])


def rxpy_cpy_imm_16(data, dst_vp: int, msk_vp: int) -> None:
    seu_layer = SEULayer.context()
    dst_rp = data.draw(st_rn_reg())
    msk_rp = data.draw(st_sm_reg())
    apl_set_rn_reg(dst_rp, dst_vp)
    apl_set_sm_reg(msk_rp, msk_vp)
    cpy_imm_16(tgt=dst_rp, val=msk_rp)


@belex_apl
def read_src_into_rsp2k(Belex, src: VR):
    RL[::] <= src()
    RSP16[::] <= RL()
    RSP256() <= RSP16()
    RSP2K() <= RSP256()
    RSP32K() <= RSP2K()
    NOOP()
    NOOP()
    RSP_END()


def rxpy_read_src_into_rsp2k(data, dst_vp: int) -> None:
    seu_layer = SEULayer.context()
    dst_rp = data.draw(st_rn_reg())
    apl_set_rn_reg(dst_rp, dst_vp)
    read_src_into_rsp2k(src=dst_rp)


@diri_subscription_test(dst_vp=st_rn_val,
                        msk_vp=st_sm_val)
def test_subscribe_to_cpy_imm_16(data,
                                 seu_layer: SEULayer,
                                 apuc_rsp_fifo: ApucRspFifo,
                                 diri: DIRI,
                                 dst_vp: int,
                                 msk_vp: int) -> None:

    # NOTE: We only want the side-effects of calling these functions
    rxpy_cpy_imm_16(data, dst_vp, msk_vp)
    rxpy_read_src_into_rsp2k(data, dst_vp)
    for apc_id in range(2):
        apuc_rsp_fifo.rsp_rd(apc_id)


def rxpy_store_16(data, vm_reg: int, src_vp: int) -> None:
    parity_grp, parity_dst_vp, dst_vp = \
        belex_gal_vm_reg_to_set_ext(vm_reg)
    parity_mask_vp = store_16_parity_mask(parity_grp)

    src_rp = data.draw(st_rn_reg())
    dst_rp = data.draw(st_l1_reg())
    parity_dst_rp = data.draw(st_l1_reg(exclude={dst_rp.register}))
    parity_mask_rp = data.draw(st_sm_reg())

    apl_set_rn_reg(src_rp, src_vp)
    apl_set_l1_reg(parity_dst_rp, parity_dst_vp)
    apl_set_l1_reg(dst_rp, dst_vp)
    apl_set_sm_reg(parity_mask_rp, parity_mask_vp)

    store_16_t0(dst=dst_rp,
                parity_dst=parity_dst_rp,
                parity_mask=parity_mask_rp,
                src=src_rp)


def belex_gal_encode_l2_addr(byte_idx: int, bit_idx: int) -> int:
    return (byte_idx << BELEX_L2_CTL_ROW_ADDR_BIT_IDX_BITS) | bit_idx


def belex_bank_group_row_to_addr(bank_id: int,
                                 group_id: int,
                                 row_id: int) -> int:
    return (bank_id << 11) | (group_id << 9) | row_id


def _copy_N_l1_to_l2(data,
                     l1_bank_id: int,
                     vm_reg: int,
                     l1_grp: int,
                     num_bytes: int,
                     l2_ready_set: bool,
                     l2_start_byte: int) -> None:
    l1_parity_grp, l1_parity_row, l1_grp_row = \
        belex_gal_vm_reg_to_set_ext(vm_reg)

    for i in range(num_bytes):
        if l1_grp >= GSI_L1_NUM_GRPS:
            l1_grp = 0
            vm_reg += 1
            l1_parity_grp, l1_parity_row, l1_grp_row = \
                belex_gal_vm_reg_to_set_ext(vm_reg)

        l2_addr = belex_gal_encode_l2_addr(l2_start_byte + i, 0)

        src_addr = \
            belex_bank_group_row_to_addr(l1_bank_id, l1_grp, l1_grp_row)

        parity_src_addr = \
            belex_bank_group_row_to_addr(l1_bank_id,
                                         l1_parity_grp,
                                         l1_parity_row)

        rxpy_copy_l1_to_l2_byte(data,
                                l2_addr,
                                src_addr,
                                parity_src_addr)

        l1_grp += 2
        l1_parity_grp += 2

    if l2_ready_set:
        l2_end()


def belex_store_vmr_16(data,
                       vm_reg: int,
                       l1_bank_id: int,
                       l2_ready_set: bool,
                       l2_start_byte: int) -> None:
    _copy_N_l1_to_l2(data,
                     l1_bank_id, vm_reg, 0, 2, l2_ready_set, l2_start_byte)


def belex_dma_l1_to_l2(data, vmr: int) -> None:
    # belex_l2dma_l2_ready_rst_all();
    for bank in range(4):
        belex_store_vmr_16(data, vmr, bank, bank == (4 - 1), (bank * 2))


def rxpy_copy_l1_to_l2_byte(data,
                            dst_vp: int,
                            src_vp: int,
                            parity_src_vp: int) -> None:

    dst_rp = data.draw(st_l2_reg())
    src_rp = data.draw(st_l1_reg())
    parity_src_rp = data.draw(st_l1_reg(exclude={src_rp.register}))

    apl_set_l2_reg(dst_rp, dst_vp)
    apl_set_l1_reg(src_rp, src_vp)
    apl_set_l1_reg(parity_src_rp, parity_src_vp)

    copy_l1_to_l2_byte(dst=dst_rp,
                       src=src_rp,
                       parity_src=parity_src_rp)


@diri_subscription_test(row_number=st_rn_val,
                        vr_data=st_vr_data(),
                        vmr=st_vmr_val)
def test_vr_to_l2(data,
                  seu_layer: SEULayer,
                  apuc_rsp_fifo: ApucRspFifo,
                  diri: DIRI,
                  row_number: int,
                  vr_data: np.ndarray,
                  vmr: int) -> None:
    diri.hb[row_number] = vr_data
    rxpy_store_16(data, vmr, row_number)
    belex_dma_l1_to_l2(data, vmr)


@belex_apl
def copy_re_to_ewe_w_rwinh(Belex, dsts: EWE, srcs: RE, mrk: VR, msk: Mask):
    RL[~msk] <= 0
    RWINH_SET[RL[msk] <= mrk()]
    RL[::] <= srcs()
    dsts[::] <= RL()
    RWINH_RST[msk]



def rxpy_copy_re_to_ewe_w_rwinh(data,
                                dsts_vp: int,
                                srcs_vp: int,
                                mrk_vp: int,
                                msk_vp: int) -> None:

    dsts_rp = data.draw(st_ewe_reg())
    srcs_rp = data.draw(st_re_reg())
    mrk_rp = data.draw(st_rn_reg())
    msk_rp = data.draw(st_sm_reg())

    apl_set_ewe_reg(dsts_rp, dsts_vp)
    apl_set_re_reg(srcs_rp, srcs_vp)
    apl_set_rn_reg(mrk_rp, mrk_vp)
    apl_set_sm_reg(msk_rp, msk_vp)

    copy_re_to_ewe_w_rwinh(dsts=dsts_rp,
                           srcs=srcs_rp,
                           mrk=mrk_rp,
                           msk=msk_rp)


@diri_subscription_test(dsts_vp=st_ewe_val,
                        srcs_vp=st_re_val,
                        mrk_vp=st_rn_val,
                        mrk_data=st_vr_data(),
                        msk_vp=st_sm_val)
def test_copy_re_to_ewe_w_rwinh(data,
                                seu_layer: SEULayer,
                                apuc_rsp_fifo: ApucRspFifo,
                                diri: DIRI,
                                dsts_vp: int,
                                srcs_vp: int,
                                mrk_vp: int,
                                mrk_data: np.ndarray,
                                msk_vp: int) -> None:
    diri.hb[mrk_vp] = mrk_data
    rxpy_copy_re_to_ewe_w_rwinh(data, dsts_vp, srcs_vp, mrk_vp, msk_vp)


@belex_apl
def copy_row_over_reg(Belex, vr: VR, msk: Mask):
    RL[::] <= vr()
    GL[msk] <= RL()
    vr[::] <= GL()


def rxpy_copy_row_over_reg(data, vr_vp: int, msk_vp: int) -> None:
    vr_rp = data.draw(st_rn_reg())
    msk_rp = data.draw(st_sm_reg())

    apl_set_rn_reg(vr_rp, vr_vp)
    apl_set_sm_reg(msk_rp, msk_vp)

    copy_row_over_reg(vr=vr_rp, msk=msk_rp)


@diri_subscription_test(vr_vp=st_rn_val,
                        vr_data=st_vr_data(),
                        msk_vp=st_sm_val)
def test_copy_row_over_reg(data,
                           seu_layer: SEULayer,
                           apuc_rsp_fifo: ApucRspFifo,
                           diri: DIRI,
                           vr_vp: int,
                           vr_data: int,
                           msk_vp: int) -> None:
    diri.hb[vr_vp] = vr_data
    rxpy_copy_row_over_reg(data, vr_vp, msk_vp)
