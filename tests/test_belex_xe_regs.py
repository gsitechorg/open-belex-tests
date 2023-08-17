r"""
By Dylon Edwards
"""

from copy import deepcopy
from typing import Sequence

import numpy as np

import hypothesis
import hypothesis.strategies as st
from hypothesis import given

from open_belex.common.constants import NSB, NSECTIONS
from open_belex.diri.half_bank import DIRI
from open_belex.literal import EWE, RE, RL, VR, Mask, belex_apl, u16
from open_belex.utils.example_utils import u16_to_bool

from open_belex_libs.common import cpy_imm_16

from open_belex_tests.utils import Mask_strategy, parameterized_belex_test

AVAILABLE_SBS = np.arange(NSB)


# Redefinition of sb_strategy that avoids SB=23
sb_strategy = st.integers(
    min_value=0,
    max_value=22)  # IMPORTANT: Pending further investigation, avoid SB=23


@st.composite
def re_strategy(
        draw: st.DrawFn,
        min_size: int = 1,
        max_size: int = 16,
        never_sample: int = -1,
        always_sample: int = -1) -> np.ndarray:

    assert min_size <= max_size

    seed = draw(st.integers(min_value=0, max_value=2**32-1))
    random = np.random.RandomState(seed)

    num_sbs = draw(st.integers(min_size, max_size))

    samples = random.choice(AVAILABLE_SBS,
                            size=num_sbs,
                            replace=False)

    if never_sample >= 0 and never_sample in samples:
        if len(samples) > 1:
            samples = np.delete(samples, samples == never_sample)
        else:
            samples[0] = NSB - samples[0] - 1

    if always_sample >= 0 and always_sample not in samples:
        samples = np.append(samples, always_sample)

    if len(samples) > num_sbs:
        samples = samples[-num_sbs:]

    return samples


def sbs_to_row_mask(sbs: np.ndarray) -> int:
    row_mask = 0x000000
    for sb in sbs:
        row_mask |= (0x000001 << sb)
    return row_mask


def row_mask_to_sbs(row_mask: int) -> Sequence[int]:
    sbs = [sb for sb in range(NSB) if row_mask & (1 << sb) != 0]
    return np.array(sbs)


@given(srcs=re_strategy())
def test_invertibility_of_sbs_to_row_mask(srcs: np.ndarray):
    row_mask = sbs_to_row_mask(srcs)
    expected_value = np.sort(srcs)
    actual_value = row_mask_to_sbs(row_mask)
    assert np.array_equal(expected_value, actual_value)


@st.composite
def ewe_strategy(
        draw: st.DrawFn,
        min_size: int = 1,
        max_size: int = 8,
        never_sample: int  = -1,
        always_sample: int = -1) -> np.ndarray:

    assert min_size <= max_size

    seed = draw(st.integers(min_value=0, max_value=2**32-1))
    random = np.random.RandomState(seed)

    num_sbs = draw(st.integers(min_size, max_size))
    num_sbs = min(num_sbs, NSB)
    assert num_sbs >= min_size

    group = draw(st.integers(0, 2))
    group_lower = group * 8
    group_upper = group_lower + 8

    samples = random.choice(AVAILABLE_SBS[group_lower:group_upper],
                            size=num_sbs,
                            replace=False)

    if never_sample >= 0 and never_sample in samples:
        if len(samples) > 1:
            samples = np.delete(samples, samples == never_sample)
        else:
            samples[0] = group_upper - samples[0] - 1

    if always_sample >= 0 \
       and always_sample not in samples \
       and group_lower <= always_sample < group_upper:
        # Only include "always_sample" if it belongs in the chosen group.
        # ^^^ This is in exception to the name of the variable.
        samples = np.append(samples, always_sample)

    if len(samples) > num_sbs:
        samples = samples[-num_sbs:]

    return samples


def sbs_to_wordline_mask(sbs: np.ndarray) -> int:
    group = sbs[0] // 8
    offset = group * 8
    wordline_mask = group << 8
    for sb in sbs:
        wordline_mask |= 1 << (sb - offset)
    return wordline_mask


def wordline_mask_to_sbs(wordline_mask: int) -> Sequence[int]:
    group = wordline_mask >> 8
    offset = group * 8
    sbs = [sb + offset for sb in range(8)
           if wordline_mask & (1 << sb) != 0]
    return np.array(sbs)


@given(srcs=ewe_strategy())
def test_invertibility_of_sbs_to_wordline_mask(srcs: np.ndarray):
    wordline_mask = sbs_to_wordline_mask(srcs)
    expected_value = np.sort(srcs)
    actual_value = wordline_mask_to_sbs(wordline_mask)
    assert np.array_equal(expected_value, actual_value)


@belex_apl
def vr_from_re(Belex, out: VR, srcs: RE):
    RL[::] <= srcs()
    out[::] <= RL()


# IMPORTANT: Pending further investigation, avoid SB=23
@hypothesis.settings(max_examples=5, deadline=None)
@given(srcs=re_strategy(never_sample=23))
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_vr_from_re(diri: DIRI, srcs: np.ndarray) -> int:
    out = 0
    row_mask = sbs_to_row_mask(srcs)
    expected_value = np.logical_and.reduce(diri.hb[srcs], axis=0)
    vr_from_re(out, row_mask)
    assert np.array_equal(expected_value, diri.hb[out])
    return out


@belex_apl
def vr_from_shifted_re(Belex, out: VR, srcs: RE):
    RL[::] <= srcs() << 3
    out[::] <= RL()


# IMPORTANT: Pending further investigation, avoid SB=23
@hypothesis.settings(max_examples=5, deadline=None)
@given(srcs=re_strategy(never_sample=20))
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_vr_from_shifted_re(diri: DIRI, srcs: np.ndarray) -> int:
    out = 0
    row_mask = sbs_to_row_mask(srcs)
    shifted_srcs = srcs + 3
    shifted_srcs = shifted_srcs[shifted_srcs < NSB]
    expected_value = np.logical_and.reduce(diri.hb[shifted_srcs], axis=0)
    vr_from_shifted_re(out, row_mask)
    assert np.array_equal(expected_value, diri.hb[out])
    return out


@belex_apl
def vr_from_negated_re(Belex, out: VR, srcs: RE):
    RL[::] <= ~srcs()
    out[::] <= RL()


# IMPORTANT: Pending further investigation, avoid SB=23
@hypothesis.settings(max_examples=5, deadline=None)
@given(srcs=re_strategy(min_size=(NSB - 16),
                        max_size=(NSB - 1),
                        always_sample=23))
@hypothesis.example(srcs=np.arange(NSB))
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_vr_from_negated_re(diri: DIRI, srcs: np.ndarray) -> int:
    out = 0
    row_mask = sbs_to_row_mask(srcs)
    negated_srcs = sorted(set(range(NSB)) - set(srcs))
    expected_value = np.logical_and.reduce(diri.hb[negated_srcs], axis=0)
    vr_from_negated_re(out, row_mask)
    assert np.array_equal(expected_value, diri.hb[out])
    return out


@belex_apl
def vr_from_shifted_and_negated_re(Belex, out: VR, srcs: RE):
    RL[::] <= ~(srcs() << 3)
    out[::] <= RL()


# IMPORTANT: Pending further investigation, avoid SB=23
@hypothesis.settings(max_examples=5, deadline=None)
@given(srcs=re_strategy(min_size=(NSB - 13),
                        max_size=(NSB - 1),
                        always_sample=20))
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_vr_from_shifted_and_negated_re(diri: DIRI, srcs: np.ndarray) -> int:
    out = 0
    row_mask = sbs_to_row_mask(srcs)
    shifted_srcs = srcs + 3
    shifted_srcs = shifted_srcs[shifted_srcs < NSB]
    shifted_and_negated_srcs = sorted(set(range(NSB)) - set(shifted_srcs))
    expected_value = np.logical_and.reduce(
        diri.hb[shifted_and_negated_srcs], axis=0)
    vr_from_shifted_and_negated_re(out, row_mask)
    assert np.array_equal(expected_value, diri.hb[out])
    return out


@belex_apl
def ewe_from_src(Belex, outs: EWE, src: VR):
    RL[::] <= src()
    outs[::] <= RL()


# IMPORTANT: Pending further investigation, avoid SB=23
@hypothesis.settings(max_examples=5, deadline=None)
@given(outs=ewe_strategy(never_sample=23),
       src=sb_strategy)
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_ewe_from_src(diri: DIRI, outs: np.ndarray, src: int) -> int:
    wordline_mask = sbs_to_wordline_mask(outs)
    expected_value = deepcopy(diri.hb[src])
    ewe_from_src(wordline_mask, src)
    for out in outs:
        assert np.array_equal(expected_value, diri.hb[out])
    if len(outs) > 0:
        return outs[0]
    return 0


@belex_apl
def shifted_ewe_from_src(Belex, outs: EWE, src: VR):
    RL[::] <= src()
    outs[::] << 1 <= RL()


# IMPORTANT: Pending further investigation, avoid SB=23
@hypothesis.settings(max_examples=5, deadline=None)
@given(outs=ewe_strategy(never_sample=22),
       src=sb_strategy)
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_shifted_ewe_from_src(diri: DIRI, outs: np.ndarray, src: int) -> int:
    wordline_mask = sbs_to_wordline_mask(outs)
    group = outs[0] // 8
    upper_bound = (1 + group) * 8
    shifted_outs = [out + 1 for out in outs if out + 1 < upper_bound]
    expected_value = deepcopy(diri.hb[src])
    shifted_ewe_from_src(wordline_mask, src)
    for out in shifted_outs:
        assert np.array_equal(expected_value, diri.hb[out])
    if len(shifted_outs) > 0:
        return shifted_outs[0]
    return 0


@belex_apl
def negated_ewe_from_src(Belex, outs: EWE, src: VR):
    RL[::] <= src()
    ~outs[::] <= RL()


# IMPORTANT: Pending further investigation, avoid SB=23
@hypothesis.settings(max_examples=5, deadline=None)
@given(outs=ewe_strategy(always_sample=23),
       src=sb_strategy)
@hypothesis.example(outs=np.arange(8),
                    src=22)
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_negated_ewe_from_src(diri: DIRI, outs: np.ndarray, src: int) -> int:
    wordline_mask = sbs_to_wordline_mask(outs)
    group = outs[0] // 8
    lower_bound = group * 8
    upper_bound = (1 + group) * 8
    negated_outs = sorted(set(range(lower_bound, upper_bound)) - set(outs))
    expected_value = deepcopy(diri.hb[src])
    negated_ewe_from_src(wordline_mask, src)
    for out in negated_outs:
        assert np.array_equal(expected_value, diri.hb[out])
    if len(negated_outs) > 0:
        return negated_outs[0]
    return 0


@belex_apl
def shifted_and_negated_ewe_from_src(Belex, outs: EWE, src: VR):
    RL[::] <= src()
    ~(outs[::] << 1) <= RL()


# IMPORTANT: Pending further investigation, avoid SB=23
@hypothesis.settings(max_examples=5, deadline=None)
@given(outs=ewe_strategy(always_sample=22),
       src=sb_strategy)
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_shifted_and_negated_ewe_from_src(diri: DIRI,
                                          outs: np.ndarray,
                                          src: int) -> int:
    wordline_mask = sbs_to_wordline_mask(outs)
    group = outs[0] // 8
    lower_bound = group * 8
    upper_bound = (1 + group) * 8
    shifted_outs = [out + 1 for out in outs if out + 1 < upper_bound]
    shifted_and_negated_outs = \
        sorted(set(range(lower_bound, upper_bound)) - set(shifted_outs))
    expected_value = deepcopy(diri.hb[src])
    shifted_and_negated_ewe_from_src(wordline_mask, src)
    for out in shifted_and_negated_outs:
        assert np.array_equal(expected_value, diri.hb[out])
    if len(shifted_and_negated_outs) > 0:
        return shifted_and_negated_outs[0]
    return 0


@belex_apl
def ewe_with_negated_mask_from_src(Belex, outs: EWE, src: VR, msk: Mask):
    RL[::] <= src()
    outs[~msk] <= RL()


# IMPORTANT: Pending further investigation, avoid SB=23
@hypothesis.settings(max_examples=5, deadline=None)
@given(outs=ewe_strategy(never_sample=23),
       src=sb_strategy,
       msk=Mask_strategy())
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_ewe_with_negated_mask_from_src(diri: DIRI,
                                        outs: np.ndarray,
                                        src: int,
                                        msk: int) -> int:
    wordline_mask = sbs_to_wordline_mask(outs)

    expected_values = [deepcopy(diri.hb[out]) for out in outs]
    for s in range(NSECTIONS):
        if (1 << s) & (~msk) != 0:
            wordline = diri.hb[src, ::, s]
            for expected_value in expected_values:
                expected_value[::, s] = wordline

    ewe_with_negated_mask_from_src(wordline_mask, src, msk)
    for out, expected_value in zip(outs, expected_values):
        assert np.array_equal(expected_value, diri.hb[out])
    if len(outs) > 0:
        return outs[0]
    return 0


@belex_apl
def negated_ewe_with_negated_mask_from_src(Belex, outs: EWE, src: VR, msk: Mask):
    RL[::] <= src()
    ~outs[~msk] <= RL()


# IMPORTANT: Pending further investigation, avoid SB=23
@hypothesis.settings(max_examples=5, deadline=None)
@given(outs=ewe_strategy(always_sample=23),
       src=sb_strategy,
       msk=Mask_strategy())
@parameterized_belex_test(repeatably_randomize_half_bank=True)
def test_negated_ewe_with_negated_mask_from_src(diri: DIRI,
                                                outs: np.ndarray,
                                                src: int,
                                                msk: int) -> int:

    wordline_mask = sbs_to_wordline_mask(outs)
    group = outs[0] // 8
    lower_bound = group * 8
    upper_bound = (1 + group) * 8
    negated_outs = sorted(set(range(lower_bound, upper_bound)) - set(outs))

    negated_msk = 0xFFFF - msk

    expected_values = [deepcopy(diri.hb[out])
                       for out in negated_outs]

    for s in range(NSECTIONS):
        if (1 << s) & negated_msk != 0:
            wordline = diri.hb[src, ::, s]
            for expected_value in expected_values:
                expected_value[::, s] = wordline

    negated_ewe_with_negated_mask_from_src(wordline_mask, src, msk)
    for out, expected_value in zip(negated_outs, expected_values):
        assert np.array_equal(expected_value, diri.hb[out])

    if len(negated_outs) > 0:
        return negated_outs[0]

    return 0


@belex_apl
def frag_w_implicit_re_reg(Belex, out: VR, val1: u16, val2: u16, val3: u16, val4: u16):
    tmp1 = Belex.VR()
    cpy_imm_16(tmp1, val1)

    tmp2 = Belex.VR()
    cpy_imm_16(tmp2, val2)

    tmp3 = Belex.VR()
    cpy_imm_16(tmp3, val3)

    tmp4 = Belex.VR()
    cpy_imm_16(tmp4, val4)

    RL[::] <= tmp1() & tmp2() & tmp3() & tmp4()
    out[::] <= RL()


@hypothesis.settings(max_examples=5, deadline=None)
@given(val1=Mask_strategy(),
       val2=Mask_strategy(),
       val3=Mask_strategy(),
       val4=Mask_strategy())
@parameterized_belex_test(repeatably_randomize_half_bank=True,
                          features={
                              "coalesce-compatible-temporaries": False,
                          })
def test_frag_w_implicit_re_reg(diri: DIRI, val1: int, val2: int, val3: int, val4: int) -> int:
    out = 0
    expected_value = u16_to_bool(val1 & val2 & val3 & val4)
    frag_w_implicit_re_reg(out, val1, val2, val3, val4)
    actual_value = diri.hb[out]
    assert np.array_equal(expected_value, actual_value)
    return out
