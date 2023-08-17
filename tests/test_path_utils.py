r"""
By Dylon Edwards and Brian Beckman
"""

import filecmp
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from open_belex.utils.path_utils import (copy_file_wrt_root,
                                         copy_tree_wrt_root, exists_wrt_root,
                                         path_wrt_root, project_root, user_tmp)


def test_project_root():
    assert Path(project_root(), ".git").is_dir()


def test_path_wrt_root():
    assert path_wrt_root(".") == project_root()
    for child in project_root().iterdir():
        assert child == path_wrt_root(child.name)


def test_copy_file_wrt_root():
    with NamedTemporaryFile(prefix="path_utils-", suffix=".py") as tmp_path_utils:
        copy_file_wrt_root(__file__, tmp_path_utils.name)
        assert filecmp.cmp(__file__, tmp_path_utils.name)


def compare_tree(source, destination):
    target = Path(destination, source.name)
    if source.is_dir():
        assert target.is_dir()
        for child in source.iterdir():
            compare_tree(child, target)
    elif source.is_file():
        assert target.is_file()
    elif source.is_symlink():
        assert target.is_symlink()


def test_copy_tree_wrt_root():
    with TemporaryDirectory() as tmpdir:
        copy_tree_wrt_root("src", tmpdir)
        compare_tree(path_wrt_root("src"), tmpdir)


def test_exists_wrt_root():
    assert exists_wrt_root(".project-root")
    assert not exists_wrt_root("test_utils_path_utils.py")


# I'm not sure how to test this otherwise and am open to suggestions
def test_user_tmp_does_not_throw_error():
    user_tmp()
