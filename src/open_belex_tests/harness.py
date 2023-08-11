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

from typing import Callable

from open_belex.bleir.types import CallerMetadata, Snippet, SnippetMetadata
from open_belex.bleir.virtual_machines import BLEIRVirtualMachine
from open_belex.decorators import build_examples, collect_actual_values
from open_belex.utils.path_utils import user_tmp


def render_bleir(name: str, fn: Callable,
                 optimizations=None,
                 base_dir=user_tmp() / "belex",
                 interpret=True):

    kwargs = {}
    if optimizations is not None:
        kwargs["optimizations"] = optimizations

    fragment_caller = fn(**kwargs)
    examples = build_examples(fragment_caller)
    actual_values = collect_actual_values(examples)
    fragment_caller_call = fragment_caller(*actual_values)
    output_dir = base_dir / name / "build"

    metadata = {
        SnippetMetadata.HEADER_FILE: f"{name}-funcs.apl.h",
        SnippetMetadata.SOURCE_FILE: f"{name}-funcs.apl",
    }

    snippet = Snippet(
        name=name,
        examples=examples,
        calls=[fragment_caller_call],
        metadata=metadata)

    vm = BLEIRVirtualMachine(output_dir=output_dir,
                             interpret=interpret)

    snippet = vm.compile(snippet)

    try:
        vm.assert_no_interpreter_failures()
    except AssertionError as error:
        if not fragment_caller.has_metadata(CallerMetadata.SHOULD_FAIL, True):
            raise error

    return snippet
