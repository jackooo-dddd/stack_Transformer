import functools
from typing import Mapping

import random
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task


# Helpers for Dyck-1 on an arbitrary symbol-pair (open_sym, close_sym)
def _random_dyck_pair(length: int, open_sym: int, close_sym: int) -> jnp.ndarray:
    """Generate a random Dyck-1 sequence of total length `length`,
    where symbols are encoded as open_sym/close_sym."""
    assert length % 2 == 0, "Length must be even per bracket-type"
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
            seq.append(open_sym)
            opens -= 1
            depth += 1
        else:
            seq.append(close_sym)
            closes -= 1
            depth -= 1
    return jnp.array(seq, dtype=jnp.int32)


def _is_valid_dyck_pair(seq: jnp.ndarray, open_sym: int, close_sym: int) -> bool:
    """Check Dyck-1 validity on the subsequence of seq selecting only open_sym/close_sym."""
    depth = 0
    for x in seq.tolist():
        if x == open_sym:
            depth += 1
        elif x == close_sym:
            depth -= 1
        else:
            continue
        if depth < 0:
            return False
    return depth == 0


class Shuffle2(task.GeneralizationTask):
    """Decide membership in Shuffle-2 = Dyck-1_{()}  ‖  Dyck-1_{[]},
       shuffled interleavings of balanced () and balanced [] strings."""

    def sample_batch(self, rng: jnp.ndarray, batch_size: int, length: int) -> Mapping[str, jnp.ndarray]:
        # Ensure length is multiple of 4 so each bracket-type has an even subsequence length
        if length % 4 != 0:
            length += (4 - (length % 4))
        half = length // 2  # total symbols per bracket-type
        num_pos = batch_size // 2
        num_neg = batch_size - num_pos
        rng_pos, rng_neg, rng_perm = jrandom.split(rng, 3)

        # --- Generate positive examples ---
        seen_pos = set()
        pos_list = []
        attempts = 0
        while len(pos_list) < num_pos and attempts < num_pos * 10:
            # draw one Dyck for () and one for []
            par = _random_dyck_pair(half, open_sym=0, close_sym=1)
            sq = _random_dyck_pair(half, open_sym=2, close_sym=3)
            # shuffle their interleaving
            idx = random.sample(range(length), half)
            idx_set = set(idx)
            merged = []
            i_par = i_sq = 0
            for pos in range(length):
                if pos in idx_set:
                    merged.append(int(par[i_par]))
                    i_par += 1
                else:
                    merged.append(int(sq[i_sq]))
                    i_sq += 1
            key = tuple(merged)
            if key not in seen_pos:
                seen_pos.add(key)
                pos_list.append(jnp.array(merged, dtype=jnp.int32))
            attempts += 1

        # --- Generate hard-negative examples ---
        hard_neg_list = []
        for arr in pos_list:
            # pick exactly one of the two bracket-types to break
            bad = random.randrange(2)
            o_sym, c_sym = (0,1) if bad == 0 else (2,3)
            seq = list(arr.tolist())
            idxs = [i for i, x in enumerate(seq) if x in (o_sym, c_sym)]
            if not idxs:
                continue
            # flip one bracket in that projection
            i = random.choice(idxs)
            seq[i] = o_sym if seq[i] == c_sym else c_sym
            hard_neg_list.append(jnp.array(seq, dtype=jnp.int32))

        # filter to ensure exactly one projection fails
        seen_hard = set()
        hard_final = []
        for seq in hard_neg_list:
            t = tuple(seq.tolist())
            if t not in seen_hard and len(hard_final) < num_neg:
                seen_hard.add(t)
                valid_count = (_is_valid_dyck_pair(seq, 0, 1)
                               + _is_valid_dyck_pair(seq, 2, 3))
                if valid_count == 1:  # one of the two is still valid, so exactly one fails
                    hard_final.append(seq)

        neg_list = hard_final

        # Stack, label, shuffle
        pos = jnp.stack(pos_list) if pos_list else jnp.empty((0, length), jnp.int32)
        neg = jnp.stack(neg_list) if neg_list else jnp.empty((0, length), jnp.int32)
        strings = jnp.vstack([pos, neg])
        labels = jnp.concatenate([
            jnp.ones(pos.shape[0], dtype=jnp.int32),
            jnp.zeros(neg.shape[0], dtype=jnp.int32)
        ])

        perm = jrandom.permutation(rng_perm, strings.shape[0])
        strings, labels = strings[perm], labels[perm]

        """SOME EXAMPLES:[()(]()[][]) -> in
        ][)([[]([))) -> out
        ]][]]])([)[] -> out
        ([[)([)(]]]) -> in"""
        # Human-readable printout: sequence form and membership
        # seqs = strings.tolist()
        # labs = labels.tolist()
        # print("Batch samples:")
        # for seq, lab in zip(seqs, labs):
        #     # map symbols back to brackets
        #     br = ''.join({0:'(',1:')',2:'[',3:']'}[x] for x in seq)
        #     t = 'in' if lab == 1 else 'out'
        #     print(f"{br} -> {t}")

        # final dedup of (string,label) pairs
        seen_final = set()
        filt_str, filt_lbl = [], []
        for s, l in zip(strings.tolist(), labels.tolist()):
            pair = (tuple(s), int(l))
            if pair not in seen_final:
                seen_final.add(pair)
                filt_str.append(s)
                filt_lbl.append(l)

        one_hot_in = jnn.one_hot(jnp.array(filt_str), num_classes=4)
        one_hot_out = jnn.one_hot(jnp.array(filt_lbl), num_classes=2)

        return {"input": one_hot_in, "output": one_hot_out}

    @property
    def input_size(self) -> int:
        return 4

    @property
    def output_size(self) -> int:
        return 2


