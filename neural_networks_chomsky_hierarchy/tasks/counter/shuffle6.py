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
            choice = 'open' if random.random() < opens/(opens+closes) else 'close'
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


class Shuffle6(task.GeneralizationTask):
    """Shuffle‑6: interleaving of six Dyck‑1 bracket languages.
       Symbol mapping: 0='(',1=')',2='[',3=']',4='{',5='}',6='<',7='>',
                       8='⟨',9='⟩',10='⌈',11='⌉'"""

    def sample_batch(self, rng: jnp.ndarray, batch_size: int, length: int) -> Mapping[str, jnp.ndarray]:
        """
        rng: Random number generator
        batch_size: How many samples we have to generate.
        The batch_size need to be even since we want exactly half of positive and negative samples.
        length: The length of the sequence we need to generate.
        """
        assert batch_size % 2 == 0, "batch_size must be even"
        half = batch_size // 2
        # round length up to multiple of 12, as we want all type of brackets to participate.
        # 12 also makes the generation of each type of bracket more convenient.
        if length % 12 != 0:
            length += (12 - (length % 12))
        length_each_type = length // 6
        rng_pos, rng_neg, rng_perm = jrandom.split(rng, 3)
        bracket_types = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11)]

        # 1) Generate positive examples by interleaving six Dyck‑1 sequences
        pos_list = []
        while len(pos_list) < half:
            # generate 6 valid Dyck sequences for each type of bracket.
            dycks = [_random_dyck_pair(length_each_type, o, c) for (o, c) in bracket_types]
            # shuffle and merge
            rng_pos, rng_merge = jrandom.split(rng_pos)
            perm = jrandom.permutation(rng_merge, length).tolist()
            merged = [None] * length
            # k is the index, seq is the corresponding sequence.
            # Writes each pair of bracket into random positions of array merged, but will maintain the relative order
            # in the bracket pair.
            for k, seq in enumerate(dycks):
                block = sorted(perm[k * length_each_type: (k + 1) * length_each_type])
                for idx, val in zip(block, seq.tolist()):
                    merged[idx] = val
            arr = jnp.array(merged, dtype=jnp.int32)
            if all(_is_valid_dyck_pair(arr, o, c) for (o, c) in bracket_types):
                pos_list.append(arr)

        # 2) Hard negatives: corrupt exactly one bracket type in each positive
        # We directly generates the hard negative list from the positive examples.
        hard_neg_list = []
        for arr in pos_list:
            bad_type = random.randrange(len(bracket_types))
            o_sym, c_sym = bracket_types[bad_type]
            seq = list(arr.tolist())
            # Find all indices of open and close symbols of this type
            open_idxs = [i for i, x in enumerate(seq) if x == o_sym]
            close_idxs = [i for i, x in enumerate(seq) if x == c_sym]
            # Try to find a valid pair where open precedes close
            candidate_pairs = [(i, j) for i in open_idxs for j in close_idxs if i < j]
            if not candidate_pairs:
                continue
            # Pick one valid (open, close) pair to corrupt by swapping them.
            i, j = random.choice(candidate_pairs)
            seq[i], seq[j] = seq[j], seq[i]
            hard_neg_list.append(jnp.array(seq, dtype=jnp.int32))

        neg_list = hard_neg_list.copy()

        # 3) Stack and label
        pos = jnp.stack(pos_list) if pos_list else jnp.empty((0,length), jnp.int32)
        neg = jnp.stack(neg_list) if neg_list else jnp.empty((0,length), jnp.int32)
        strings = jnp.vstack([pos, neg])
        labels  = jnp.concatenate([
            jnp.ones(pos.shape[0],  dtype=jnp.int32),
            jnp.zeros(neg.shape[0], dtype=jnp.int32)
        ])

        # 4) Shuffle batch
        perm = jrandom.permutation(rng_perm, strings.shape[0])
        strings, labels = strings[perm], labels[perm]

        # 5) Remove the duplicate data and builds (sequence, label) pairs.
        seen_f = set()
        final_s, final_l = [], []
        for s, l in zip(strings.tolist(), labels.tolist()):
            pair = (tuple(s), int(l))
            if pair not in seen_f:
                seen_f.add(pair)
                final_s.append(s)
                final_l.append(l)

        one_hot_in  = jnn.one_hot(jnp.array(final_s), num_classes=12)
        one_hot_out = jnn.one_hot(jnp.array(final_l), num_classes=2)
        """SOME EXAMPLES: <{{<>(⌈><)⟨⟨ -> out
        <⟨[⌈(⟩{])}⌉> -> in
        ⌈(⌈([{>><)(> -> out
        ⌈[{(<}])⌉>⟨⟩ -> in
        ({<{>⌈[⟨]⟩⌉) -> out"""
        # symbol_map = {
        #     0:'(',1:')',2:'[',3:']',4:'{',5:'}',6:'<',7:'>',
        #     8:'⟨',9:'⟩',10:'⌈',11:'⌉'
        # }
        # print("Batch samples:")
        # for seq, lab in zip(strings.tolist(), labels.tolist()):
        #     br = ''.join(symbol_map[x] for x in seq)
        #     print(f"{br} -> {'in' if lab==1 else 'out'}")

        return {"input": one_hot_in, "output": one_hot_out}

    @property
    def input_size(self) -> int:
        return 12

    @property
    def output_size(self) -> int:
        return 2
