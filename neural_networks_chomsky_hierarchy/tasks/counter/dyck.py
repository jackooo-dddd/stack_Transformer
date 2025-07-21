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
    depth = 0
    for x in seq.tolist():
        depth += 1 if x == 0 else -1
        if depth < 0:
            return False
    return depth == 0


class Dyck1(task.GeneralizationTask):
    """A task to decide membership in the Dyck-1 language (balanced parentheses)."""

    def sample_batch(self, rng: jnp.ndarray, batch_size: int, length: int) -> Mapping[str, jnp.ndarray]:
        if length % 2 == 1:
            length += 1

        num_balanced = batch_size // 2
        num_unbalanced = batch_size - num_balanced
        rng_bal, rng_unb, rng_perm = jrandom.split(rng, 3)

        # Generate balanced strings efficiently
        seen_balanced = set()
        balanced_list = []
        max_attempts = num_balanced * 10
        attempts = 0
        while len(balanced_list) < num_balanced and attempts < max_attempts:
            arr = _random_dyck_sequence(length)
            key = tuple(arr.tolist())
            if key not in seen_balanced:
                seen_balanced.add(key)
                balanced_list.append(arr)
            attempts += 1

        if not balanced_list:
            return {"input": jnp.empty((0, length, 2)), "output": jnp.empty((0, 2))}

        balanced = jnp.stack(balanced_list)

        # --- Hard negatives: flip exactly one bracket in a balanced sequence ---
        hard_neg_list = []
        for arr in balanced_list:
            seq = list(arr.tolist())
            # pick a random position and flip it
            i = random.randrange(length)
            seq[i] = 0 if seq[i] == 1 else 1
            neg_arr = jnp.array(seq, dtype=jnp.int32)
            if not _is_valid_dyck(neg_arr):
                hard_neg_list.append(neg_arr)

        # pick up to num_unbalanced of these hard negatives (deduped)
        seen_hard = set()
        hard_final = []
        for arr in hard_neg_list:
            key = tuple(arr.tolist())
            if key not in seen_hard and len(hard_final) < num_unbalanced:
                seen_hard.add(key)
                hard_final.append(arr)

        # if we still need more negatives, fall back to random sampling
        seen_all = set(seen_balanced)
        neg_list = hard_final.copy()
        attempts = 0
        while len(neg_list) < num_unbalanced and attempts < num_unbalanced * 10:
            py_seq = [random.choice([0, 1]) for _ in range(length)]
            arr = jnp.array(py_seq, dtype=jnp.int32)
            key = tuple(py_seq)
            if key not in seen_all and not _is_valid_dyck(arr):
                seen_all.add(key)
                neg_list.append(arr)
            attempts += 1

        # Stack, label, shuffle
        pos = jnp.stack(balanced_list) if balanced_list else jnp.empty((0, length), jnp.int32)
        neg = jnp.stack(neg_list) if neg_list else jnp.empty((0, length), jnp.int32)
        strings = jnp.vstack([pos, neg])
        labels = jnp.concatenate([
            jnp.ones(pos.shape[0], dtype=jnp.int32),
            jnp.zeros(neg.shape[0], dtype=jnp.int32)
        ])
        perm = jrandom.permutation(rng_perm, strings.shape[0])
        strings, labels = strings[perm], labels[perm]

        strings_list = strings.tolist()
        labels_list = labels.tolist()
        seen_pairs = set()
        filtered_strings = []
        filtered_labels = []
        for s, l in zip(strings_list, labels_list):
            pair = (tuple(s), l)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                filtered_strings.append(s)
                filtered_labels.append(l)

        one_hot_strings = jnn.one_hot(jnp.array(filtered_strings), num_classes=2)
        labels_onehot = jnn.one_hot(jnp.array(filtered_labels), num_classes=2)

        """SOME EXAMPLES: ())(())) -> unbalanced
        )))()((( -> unbalanced
        (()))(())())()(((()))) -> unbalanced
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

        return {"input": one_hot_strings, "output": labels_onehot}

    @property
    def input_size(self) -> int:
        return 2

    @property
    def output_size(self) -> int:
        return 2

