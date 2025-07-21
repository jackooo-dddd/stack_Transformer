import functools
from typing import Mapping

import random
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task


# Helpers for Dyck‑1 on an arbitrary symbol‑pair
def _random_dyck_pair(length: int, open_sym: int, close_sym: int) -> jnp.ndarray:
    assert length % 2 == 0, "Length must be even per bracket‑type"
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
        assert batch_size % 2 == 0, "batch_size must be even"
        half = batch_size // 2

        if length % 12 != 0:
            length += (12 - (length % 12))
        sixth = length // 6

        rng_pos, rng_neg, rng_perm = jrandom.split(rng, 3)
        bracket_types = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11)]

        # 1) positives
        seen_pos = set()
        pos_list = []
        attempts = 0
        while len(pos_list) < half and attempts < half * 300:
            dycks = [_random_dyck_pair(sixth, o, c) for (o, c) in bracket_types]
            rng_pos, rng_merge = jrandom.split(rng_pos)
            perm = jrandom.permutation(rng_merge, length).tolist()
            merged = [None] * length
            for k, seq in enumerate(dycks):
                block = sorted(perm[k * sixth: (k + 1) * sixth])
                for idx, val in zip(block, seq.tolist()):
                    merged[idx] = val
            arr = jnp.array(merged, dtype=jnp.int32)
            if all(_is_valid_dyck_pair(arr, o, c) for (o, c) in bracket_types):
                key = tuple(merged)
                if key not in seen_pos:
                    seen_pos.add(key)
                    pos_list.append(arr)
            attempts += 1

        # 2) Hard negatives for Shuffle6
        hard_neg_list = []
        for arr in pos_list:
            bad = random.randrange(len(bracket_types))
            o_sym, c_sym = bracket_types[bad]
            seq = list(arr.tolist())
            idxs = [i for i, x in enumerate(seq) if x in (o_sym, c_sym)]
            if not idxs:
                continue
            i = random.choice(idxs)
            seq[i] = o_sym if seq[i] == c_sym else c_sym
            hard_neg_list.append(jnp.array(seq, dtype=jnp.int32))

        seen_hard = set()
        hard_final = []
        for seq in hard_neg_list:
            t = tuple(seq.tolist())
            if t not in seen_hard and len(hard_final) < half:
                seen_hard.add(t)
                if sum(_is_valid_dyck_pair(seq, o, c) for o, c in bracket_types) == 5:
                    hard_final.append(seq)
        neg_list = hard_final.copy()

        # 3) stack, shuffle, dedupe, one-hot
        pos = jnp.stack(pos_list) if pos_list else jnp.empty((0,length), jnp.int32)
        neg = jnp.stack(neg_list) if neg_list else jnp.empty((0,length), jnp.int32)
        strings = jnp.vstack([pos, neg])
        labels  = jnp.concatenate([
            jnp.ones(pos.shape[0],  dtype=jnp.int32),
            jnp.zeros(neg.shape[0], dtype=jnp.int32)
        ])
        perm = jrandom.permutation(rng_perm, strings.shape[0])
        strings, labels = strings[perm], labels[perm]

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
        return {"input": one_hot_in, "output": one_hot_out}

    @property
    def input_size(self) -> int:
        return 12

    @property
    def output_size(self) -> int:
        return 2


class Shuffle4(task.GeneralizationTask):
    """Shuffle‑4: interleaving of four Dyck‑1 bracket types.
       Symbol mapping: 0='(',1=')', 2='[',3=']', 4='{',5='}', 6='<',7='>'"""

    def sample_batch(self, rng: jnp.ndarray, batch_size: int, length: int) -> Mapping[str, jnp.ndarray]:
        assert batch_size % 2 == 0, "batch_size must be even"
        half = batch_size // 2

        if length % 8 != 0:
            length += (8 - (length % 8))
        quarter = length // 4

        rng_pos, rng_neg, rng_perm = jrandom.split(rng, 3)
        bracket_types = [(0,1),(2,3),(4,5),(6,7)]

        # 1) positives
        seen_pos = set()
        pos_list = []
        attempts = 0
        while len(pos_list) < half and attempts < half * 20:
            parts = [_random_dyck_pair(quarter, o, c) for (o, c) in bracket_types]
            rng_pos, rng_merge = jrandom.split(rng_pos)
            perm = jrandom.permutation(rng_merge, length).tolist()
            shuffled = [None] * length
            for k, part in enumerate(parts):
                block = sorted(perm[k * quarter: (k + 1) * quarter])
                for idx, sym in zip(block, part.tolist()):
                    shuffled[idx] = sym
            arr = jnp.array(shuffled, dtype=jnp.int32)
            if all(_is_valid_dyck_pair(arr, o, c) for (o, c) in bracket_types):
                key = tuple(shuffled)
                if key not in seen_pos:
                    seen_pos.add(key)
                    pos_list.append(arr)
            attempts += 1

        # 2) Hard negatives for Shuffle4
        hard_neg_list = []
        for arr in pos_list:
            bad = random.randrange(len(bracket_types))
            o_sym, c_sym = bracket_types[bad]
            seq = list(arr.tolist())
            idxs = [i for i, x in enumerate(seq) if x in (o_sym, c_sym)]
            if not idxs:
                continue
            i = random.choice(idxs)
            seq[i] = o_sym if seq[i] == c_sym else c_sym
            hard_neg_list.append(jnp.array(seq, dtype=jnp.int32))

        seen_hard = set()
        hard_final = []
        for seq in hard_neg_list:
            t = tuple(seq.tolist())
            if t not in seen_hard and len(hard_final) < half:
                seen_hard.add(t)
                if sum(_is_valid_dyck_pair(seq, o, c) for o, c in bracket_types) == 3:
                    hard_final.append(seq)
        neg_list = hard_final.copy()

        # 3) stack and label
        pos = jnp.stack(pos_list) if pos_list else jnp.empty((0, length), jnp.int32)
        neg = jnp.stack(neg_list) if neg_list else jnp.empty((0, length), jnp.int32)
        strings = jnp.vstack([pos, neg])
        labels = jnp.concatenate([
            jnp.ones(pos.shape[0], dtype=jnp.int32),
            jnp.zeros(neg.shape[0], dtype=jnp.int32)
        ])

        # 4) shuffle entire batch
        perm = jrandom.permutation(rng_perm, strings.shape[0])
        strings, labels = strings[perm], labels[perm]
        """SOME EXAMPLES: >{)}><{{<[[]}>[]][{{(][< -> out
        (<[{}>{)({<})}]>([[[]<)> -> out
        (((]}({]{>[>)(<]}}[>>(]{ -> out
        {[]<{<>(<)({(>}[])>[)}]} -> in
        <>[{()(}][{<(}>][){<>}]) -> in        """
        # # 5) human-readable example prints
        # symbol_map = {
        #     0: '(', 1: ')', 2: '[', 3: ']', 4: '{', 5: '}', 6: '<', 7: '>',
        #     8: '⟨', 9: '⟩', 10: '⌈', 11: '⌉'
        # }
        # for seq, lab in zip(strings.tolist(), labels.tolist()):
        #     br = ''.join(symbol_map[x] for x in seq)
        #     print(f"{br} -> {'in' if lab == 1 else 'out'}")

        # 6) dedupe & one-hot
        seen_f = set()
        final_s, final_l = [], []
        for s, l in zip(strings.tolist(), labels.tolist()):
            pair = (tuple(s), int(l))
            if pair not in seen_f:
                seen_f.add(pair)
                final_s.append(s)
                final_l.append(l)

        one_hot_in = jnn.one_hot(jnp.array(final_s), num_classes=8)
        one_hot_out = jnn.one_hot(jnp.array(final_l), num_classes=2)
        return {"input": one_hot_in, "output": one_hot_out}

    @property
    def input_size(self) -> int:
        return 8

    @property
    def output_size(self) -> int:
        return 2
