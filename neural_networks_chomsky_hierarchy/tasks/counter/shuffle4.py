import functools
from typing import Mapping

import random
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task

def _random_dyck_pair(length: int, open_sym: int, close_sym: int) -> jnp.ndarray:
    assert length % 2 == 0, "Length must be even per bracket‑type for dyck pair"
    half = length // 2
    opens, closes = half, half
    depth = 0
    seq = []
    while opens > 0 or closes > 0:
        if opens == 0:
            choice = 'close'
        elif closes == 0 or depth == 0:
            choice = 'open'
        else:
            choice = 'open' if random.random() < opens / (opens + closes) else 'close'
        if choice == 'open':
            seq.append(open_sym); opens -= 1; depth += 1
        else:
            seq.append(close_sym); closes -= 1; depth -= 1
    return jnp.array(seq, dtype=jnp.int32)

def _is_valid_dyck_pair(seq: jnp.ndarray, open_sym: int, close_sym: int) -> bool:
    """Give one type of bracket, check if that type of bracket is valid inside the sequence.
    NOTE that only if all types of bracket are valid inside the sequence, the sequence is valid."""
    depth = 0
    for x in seq.tolist():
        if x == open_sym:
            depth += 1
        elif x == close_sym:
            depth -= 1
        if depth < 0:
            return False
    return depth == 0

class Shuffle4(task.GeneralizationTask):
    """Shuffle‑4: interleaving of four Dyck‑1 bracket languages,
       where each bracket type has a random even-length subsequence,
       summing to the requested total length.
       Symbol mapping: 0='(',1=')',2='[',3=']',4='{',5='}',6='<',7='>'"""

    def sample_batch(self, rng: jnp.ndarray, batch_size: int, length: int) -> Mapping[str, jnp.ndarray]:
        """
        rng: Random number generator
        batch_size: How many samples we have to generate.
        The batch_size need to be even since we want exactly half of positive and negative samples.
        length: The length of the sequence we need to generate.
        """
        assert batch_size % 2 == 0, "batch_size must be even"
        half_bs = batch_size // 2

        # --- 1) Enforce minimum length of 8 and evenness ---
        if length < 8:
            length = 8
        if length % 2 != 0:
            length += 1

        # --- 2) Randomly split total_pairs = length//2 into 4 non-negative counts ---
        # In any case, the length of each pair will be even.
        total_pairs = length // 2
        if total_pairs >= 4:
            cuts = sorted(random.sample(range(1, total_pairs), 3))
            boundaries = [0] + cuts + [total_pairs]
            pair_counts = [boundaries[i+1] - boundaries[i] for i in range(4)]
        else:
            base = total_pairs // 4
            extras = total_pairs % 4
            pair_counts = [base + (1 if i < extras else 0) for i in range(4)]
        # convert to token counts (2 tokens per pair)
        br_lengths = [2 * p for p in pair_counts]
        # print(br_lengths)
        rng_pos, rng_neg, rng_perm = jrandom.split(rng, 3)
        bracket_types = [(0,1),(2,3),(4,5),(6,7)]

        # precompute index slices for each bracket type
        idx_slices = []
        start = 0
        for L in br_lengths:
            idx_slices.append((start, start + L))
            start += L

        # --- 3) Generate positive examples ---
        pos_list: list[jnp.ndarray] = []
        while len(pos_list) < half_bs:
            # generate one Dyck-1 subsequence per type
            dycks = [
                _random_dyck_pair(br_lengths[i], o, c)
                for i, (o, c) in enumerate(bracket_types)
            ]
            rng_pos, rng_merge = jrandom.split(rng_pos)
            perm = jrandom.permutation(rng_merge, length).tolist()
            merged = [None] * length

            # i is the index, seq is the corresponding sequence.
            # Writes each pair of bracket into random positions of array merged, but will maintain the relative order
            # in the bracket pair.
            for i, seq in enumerate(dycks):
                lo, hi = idx_slices[i]
                block = sorted(perm[lo:hi])
                for idx, val in zip(block, seq.tolist()):
                    merged[idx] = val

            arr = jnp.array(merged, dtype=jnp.int32)
            if all(_is_valid_dyck_pair(arr, o, c) for (o, c) in bracket_types):
                pos_list.append(arr)

        # --- 4) Generate hard negatives by swapping one matched pair ---
        # Hard negatives: corrupt exactly one bracket type in each positive
        # We directly generates the hard negative list from the positive examples.
        hard_neg_list: list[jnp.ndarray] = []
        for arr in pos_list:
            bad = random.randrange(4)
            o_sym, c_sym = bracket_types[bad]
            seq = list(arr.tolist())
            # Find all indices of open and close symbols of this type
            opens = [i for i,x in enumerate(seq) if x == o_sym]
            closes = [i for i,x in enumerate(seq) if x == c_sym]
            # Try to find a valid pair where open precedes close
            pairs = [(i,j) for i in opens for j in closes if i < j]
            if not pairs:
                continue
            # Pick one valid (open, close) pair to corrupt by swapping them.
            i, j = random.choice(pairs)
            seq[i], seq[j] = seq[j], seq[i]
            hard_neg_list.append(jnp.array(seq, dtype=jnp.int32))

        neg_list = hard_neg_list

        # --- 5) Stack, shuffle, remove duplicates, and one-hot encode ---
        pos_arr = jnp.stack(pos_list)
        neg_arr = (jnp.stack(neg_list)
                   if neg_list else jnp.empty((0, length), jnp.int32))

        strings = jnp.vstack([pos_arr, neg_arr])
        labels  = jnp.concatenate([
            jnp.ones(pos_arr.shape[0],  dtype=jnp.int32),
            jnp.zeros(neg_arr.shape[0], dtype=jnp.int32)
        ])
        permute = jrandom.permutation(rng_perm, strings.shape[0])
        strings, labels = strings[permute], labels[permute]

        seen = set()
        final_s: list[list[int]] = []
        final_l: list[int] = []
        for s, l in zip(strings.tolist(), labels.tolist()):
            key = (tuple(s), int(l))
            if key not in seen:
                seen.add(key)
                final_s.append(s)
                final_l.append(l)

        one_hot_in  = jnn.one_hot(jnp.array(final_s), num_classes=8)
        one_hot_out = jnn.one_hot(jnp.array(final_l), num_classes=2)

        """SOME EXAMPLES: >{)}><{{<[[]}>[]][{{(][< -> out
        (((]}({]{>[>)(<]}}[>>(]{ -> out
        {[]<{<>(<)({(>}[])>[)}]} -> in
        <>[{()(}][{<(}>][){<>}]) -> in        """
        # symbol_map = {
        #     0:'(',1:')',2:'[',3:']',4:'{',5:'}',6:'<',7:'>'
        # }
        # print("Batch samples:")
        # for seq, lab in zip(strings.tolist(), labels.tolist()):
        #     br = ''.join(symbol_map[x] for x in seq)
        #     print(f"{br} -> {'in' if lab==1 else 'out'}")

        return {"input": one_hot_in, "output": one_hot_out}

    @property
    def input_size(self) -> int:
        return 8

    @property
    def output_size(self) -> int:
        return 2
