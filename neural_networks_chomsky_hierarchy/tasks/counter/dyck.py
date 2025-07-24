import functools
from typing import Mapping

import random
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task

def _random_dyck_sequence(length: int) -> jnp.ndarray:
    assert length % 2 == 0, "Length must be even for Dyck-1 strings"
    half = length // 2
    opens_remain = half
    closes_remain = half
    depth = 0
    seq = []
    while opens_remain > 0 or closes_remain > 0:
        if opens_remain == 0:
            choice = 'close'
        elif closes_remain == 0 or depth == 0:
            choice = 'open'
        else:
            choice = 'open' if random.random() < opens_remain / (opens_remain + closes_remain) else 'close'
        if choice == 'open':
            seq.append(0)
            opens_remain -= 1
            depth += 1
        else:
            seq.append(1)
            closes_remain -= 1
            depth -= 1
    return jnp.array(seq, dtype=jnp.int32)

def _is_valid_dyck(seq: jnp.ndarray) -> bool:
    """Check that a sequence of 0/1 tokens is a valid balanced-parentheses (Dyck-1) string."""
    depth = 0
    for x in seq.tolist():
        depth += 1 if x == 0 else -1
        if depth < 0:
            return False
    return depth == 0

class Dyck1(task.GeneralizationTask):
    """A task to decide membership in the Dyck-1 language (balanced parentheses)."""

    def sample_batch(self, rng: jnp.ndarray, batch_size: int, length: int) -> Mapping[str, jnp.ndarray]:
        assert batch_size % 2 == 0, "batch_size must be even"
        num_pos = batch_size // 2
        num_neg = batch_size - num_pos
        rng_pos, rng_neg, rng_perm = jrandom.split(rng, 3)

        # --- 1) Enforce minimum length of 2 and evenness ---
        if length < 2:
            length = 2
        if length % 2 != 0:
            length += 1

        # --- 2) Generate balanced (positive) examples ---
        seen_pos = set()
        balanced_list: list[jnp.ndarray] = []
        attempts = 0
        while len(balanced_list) < num_pos and attempts < num_pos * 10:
            arr = _random_dyck_sequence(length)
            key = tuple(arr.tolist())
            if key not in seen_pos:
                seen_pos.add(key)
                balanced_list.append(arr)
            attempts += 1

        # --- 3) Generate hard negatives by swapping one matched pair ---
        hard_neg_list: list[jnp.ndarray] = []
        for arr in balanced_list:
            seq = list(arr.tolist())
            opens = [i for i, x in enumerate(seq) if x == 0]
            closes = [j for j, x in enumerate(seq) if x == 1]
            pairs = [(i, j) for i in opens for j in closes if i < j]
            if not pairs:
                continue
            i, j = random.choice(pairs)
            seq[i], seq[j] = seq[j], seq[i]
            neg_arr = jnp.array(seq, dtype=jnp.int32)
            if not _is_valid_dyck(neg_arr):
                hard_neg_list.append(neg_arr)

        # --- 4) Fallback to random negatives if needed ---
        neg_list = hard_neg_list.copy()
        seen_all = set(seen_pos)
        attempts = 0
        while len(neg_list) < num_neg and attempts < num_neg * 10:
            py_seq = [random.choice([0, 1]) for _ in range(length)]
            arr = jnp.array(py_seq, dtype=jnp.int32)
            key = tuple(py_seq)
            if key not in seen_all and not _is_valid_dyck(arr):
                seen_all.add(key)
                neg_list.append(arr)
            attempts += 1

        # --- 5) Stack, shuffle, dedupe, and one-hot encode ---
        pos_arr = jnp.stack(balanced_list) if balanced_list else jnp.empty((0, length), jnp.int32)
        neg_arr = jnp.stack(neg_list) if neg_list else jnp.empty((0, length), jnp.int32)
        strings = jnp.vstack([pos_arr, neg_arr])
        labels  = jnp.concatenate([
            jnp.ones(pos_arr.shape[0], dtype=jnp.int32),
            jnp.zeros(neg_arr.shape[0], dtype=jnp.int32)
        ])
        perm = jrandom.permutation(rng_perm, strings.shape[0])
        strings, labels = strings[perm], labels[perm]

        """SOME EXAMPLES: ())(())) -> unbalanced
        )))()((( -> unbalanced
        ()()(()) -> balanced
        ()(()((( -> unbalanced        """
        # Human-readable printout: bracket form and membership
        # seqs = strings.tolist()
        # labs = labels.tolist()
        # print("Batch samples:")
        # for seq, lab in zip(seqs, labs):
        #     brackets = ''.join('(' if x == 0 else ')' for x in seq)
        #     type_str = 'balanced' if lab == 1 else 'unbalanced'
        #     print(f"{brackets} -> {type_str}")

        # Final dedupe
        seen_final = set()
        filt_str, filt_lbl = [], []
        for s, l in zip(strings.tolist(), labels.tolist()):
            pair = (tuple(s), int(l))
            if pair not in seen_final:
                seen_final.add(pair)
                filt_str.append(s)
                filt_lbl.append(l)

        one_hot_in  = jnn.one_hot(jnp.array(filt_str), num_classes=2)
        one_hot_out = jnn.one_hot(jnp.array(filt_lbl), num_classes=2)
        return {"input": one_hot_in, "output": one_hot_out}

    @property
    def input_size(self) -> int:
        return 2

    @property
    def output_size(self) -> int:
        return 2
