r"""
By Dylon Edwards
"""

from open_belex.diri.half_bank import DIRI
from open_belex.literal import GGL, L1, belex_apl

from open_belex_libs.common import reset_16

from open_belex_tests.utils import parameterized_belex_test


@belex_apl
def l1_from_ggl(Belex, l1_row: L1):
    l1_row() <= GGL()


@parameterized_belex_test
def test_l1_from_ggl(diri: DIRI):
    reset_16(0)
    l1_from_ggl(0)
    return 0
