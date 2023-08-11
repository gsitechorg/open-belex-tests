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

from collections import defaultdict

from open_belex.directed_graph import DirectedGraph


def test_eq():
    g = DirectedGraph([])
    h = DirectedGraph([])
    assert g == h

    g = DirectedGraph([])
    h = DirectedGraph([1, 2])
    assert g != h

    g = DirectedGraph([1, 2])
    h = DirectedGraph([1, 2])
    assert g == h

    h.add_edge(1, 2)
    assert g != h

    g.add_edge(1, 2)
    assert g == h


def test_clone():
    g = DirectedGraph([])
    h = g.clone()
    assert g == h

    g = DirectedGraph([1, 2])
    h = g.clone()
    assert g == h

    h.add_edge(1, 2)
    assert g != h

    g.add_edge(1, 2)
    assert g == h

    g = DirectedGraph([1, 2, 3])
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(1, 3)
    h = g.clone()
    assert g == h


def test_reachable():
    g = DirectedGraph([1, 2, 3, 4, 5, 6])
    g.add_edge(1, 2)
    g.add_edge(1, 4)
    g.add_edge(2, 3)
    g.add_edge(1, 5)
    assert g.reachable(1, 2)
    assert g.reachable(1, 4)
    assert g.reachable(2, 3)
    assert g.reachable(1, 5)
    assert g.reachable(1, 3)
    assert not g.reachable(2, 1)
    assert not g.reachable(4, 5)
    assert not g.reachable(1, 6)


def test_topological_sort():
    from_deps = defaultdict(set)
    to_deps = defaultdict(set)
    g = DirectedGraph([1, 2, 3, 4, 5, 6])

    from_deps[1].add(2)
    to_deps[2].add(1)
    g.add_edge(1, 2)

    from_deps[4].add(6)
    to_deps[6].add(4)
    g.add_edge(4, 6)

    from_deps[2].add(5)
    to_deps[5].add(2)
    g.add_edge(2, 5)

    from_deps[3].add(5)
    to_deps[5].add(3)
    g.add_edge(3, 5)

    from_deps[2].add(4)
    to_deps[4].add(2)
    g.add_edge(2, 4)

    from_deps[1].add(3)
    to_deps[3].add(1)
    g.add_edge(1, 3)

    ordering = g.topological_sort()
    assert set(ordering) == g.V

    # The constraint of a topological is that no vertex may be visited before
    # its dependencies are visited. Other than that, there is no constraint on
    # order.
    for v in ordering:
        assert len(to_deps[v]) == 0
        for w in from_deps[v]:
            to_deps[w].remove(v)
        del from_deps[v]
        del to_deps[v]

    assert len(from_deps) == 0
    assert len(to_deps) == 0
