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

from typing import Optional, Sequence
from collections import deque

import numpy as np


# See: https://www.keithschwarz.com/interesting/code/?dir=alias-method
# See: https://www.keithschwarz.com/darts-dice-coins/
class AliasMethod:
    r"""An implementation of the alias method implemented using Vose's algorithm. The alias method
    allows for efficient sampling of random values from a discrete probability distribution (i.e.
    rolling a loaded die) in O(1) time each after O(n) preprocessing time.

    For a complete writeup on the alias method, including the intuition and important proofs,
    please see the article, "Darts, Dice, and Coins: Sampling from a Discrete Distribution" at

        https://www.keithschwarz.com/darts-dice-coins/
    """

    # The random number generator used to sample from the distribution
    random: np.random.RandomState

    # The probability and alias tables
    alias: np.ndarray
    probability: np.ndarray

    def __init__(self: "AliasMethod",
                 probabilities: Sequence[float],
                 random: Optional[np.random.RandomState] = None,
                 seed: Optional[int] = None):
        r"""Constructs a new AliasMethod to sample from a discrete distribution and hand back
        outcomes based on the probability distribution.

        Given as input a list of probabilities corresponding to outcomes 0, 1, ..., n - 1, along
        with the random number generator that should be used as the underlying generator, this
        constructor creates the probability and alias tables needed to efficiently sample from this
        distribution.

        Parameters:
            probabilities The list of probabilities
            random The random number generator (RNG)
            seed Seed value for constructing a default RNG
        """

        N = len(probabilities)
        assert N > 0

        self.alias = np.ndarray(N, dtype=np.int64)
        self.probability = np.ndarray(N, dtype=np.float64)

        if random is None:
            random = np.random.RandomState(seed=seed)
        self.random = random

        # Compute the average probability and cache it for later use
        average = 1.0 / N

        # Make a copy of the probabilities list, since we will be making changes to it
        probabilities = list(probabilities)

        # Create two stacks to act as worklists as we populate the tables
        small = deque()
        large = deque()

        # Populate the stacks with the input probabilities
        for i in range(N):
            if probabilities[i] < average:
                small.append(i)
            else:
                large.append(i)

        # As a note: in the mathematical specification of the algorithm, we will always exhaust the
        # small list before the big list. However, due to floating point inaccuracies, this is not
        # necessarily true. Consequently, this inner loop (which tries to pair small and large
        # elements) will have to check that both lists aren't empty.
        while len(small) > 0 and len(large) > 0:
            # Get the indices of the small and large probabilities
            less = small.pop()
            more = large.pop()

            # These probabilities have not yet been scaled up such that 1/n is given the weight
            # 1.0. We do this here instead.
            self.probability[less] = probabilities[less] * N
            self.alias[less] = more

            # Decrease the probability of the larger one by the appropriate amount
            probabilities[more] += probabilities[less] - average

            # If the new probability is less than the average, add it into the small list;
            # otherwise, add it to the large list
            if probabilities[more] < average:
                small.append(more)
            else:
                large.append(more)

        # At this point, everything is in one list, which means the remaining probabilities should
        # all be 1/n. Based on this, set them appropriately. Due to numerical issues, we can't be
        # sure which stack will hold the entries, so we empty both.
        while len(small) > 0:
            less = small.pop()
            self.probability[less] = 1.0
        while len(large) > 0:
            more = large.pop()
            self.probability[more] = 1.0

    def sample(self: "AliasMethod") -> int:
        """Samples a value from the underlying distribution.

        Returns:
            A random value sampled from the underlying distribution"""

        # Generate a fair die roll to determine which column to inspect
        column = self.random.randint(low=0, high=len(self.probability))

        # Generate a biased coin toss to determine which option to pick
        coin_toss = self.random.random_sample() < self.probability[column]

        # Based on the outcome, return either the column or its alias.
        return column if coin_toss else self.alias[column]
