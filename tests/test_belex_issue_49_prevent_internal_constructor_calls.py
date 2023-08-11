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
