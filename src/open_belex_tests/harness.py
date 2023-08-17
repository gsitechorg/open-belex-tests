r"""
By Dylon Edwards
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
