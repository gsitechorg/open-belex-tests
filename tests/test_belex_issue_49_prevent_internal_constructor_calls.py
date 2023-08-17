r"""
By Dylon Edwards
"""

import pytest

from open_belex.literal import VR, belex_apl


def test_tmp_constructors():

    @belex_apl
    def empty_frag(Belex):
        pass

    @belex_apl
    def frag_w_user_param(Belex, out: VR):
        pass

    @belex_apl
    def frag_w_temp(Belex):
        tmp = Belex.VR()

    @belex_apl
    def frag_w_user_param_and_temp(Belex, out: VR):
        tmp = Belex.VR()

    @belex_apl
    def frag_w_invalid_temp(Belex):
        invalid_tmp_throws_error = VR(0)  # <<- does not use Belex.VR constructor

    @belex_apl
    def frag_w_user_param_and_invalid_temp(Belex, out: VR):
        invalid_tmp_throws_error = VR(0)  # <<- does not use Belex.VR constructor

    out = 0

    empty_frag()
    frag_w_user_param(out)
    frag_w_temp()
    frag_w_user_param_and_temp(out)

    with pytest.raises(AssertionError):
        frag_w_invalid_temp()

    with pytest.raises(AssertionError):
        frag_w_user_param_and_invalid_temp(out)
