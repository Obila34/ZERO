"""GuestBook: cluster unfamiliar voices into stable, separate provisional ids."""
from __future__ import annotations

import numpy as np
import pytest

from zero.identity.guests import GuestBook, guest_label


def _unit(seed: int, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def book(tmp_path):
    return GuestBook(str(tmp_path / "g.sqlite"), match_threshold=0.5, max_guests=3)


def test_same_voice_same_guest_different_voice_new(book):
    a = book.assign(_unit(1))
    a_again = book.assign(_unit(1) + 0.03 * _unit(9))   # nearly identical
    b = book.assign(_unit(2))                            # clearly different
    assert a is not None and a < 0                       # guests are negative ids
    assert a_again == a                                  # clustered together
    assert b != a                                        # a separate stranger
    assert book.count() == 2


def test_two_strangers_are_not_lumped(book):
    # The exact bug this fixes: two different unknown speakers must not merge.
    ids = {book.assign(_unit(s)) for s in (10, 20, 30)}
    assert len(ids) == 3


def test_unusable_embedding_returns_none(book):
    assert book.assign(None) is None
    assert book.assign(np.zeros(32, dtype=np.float32)) is None


def test_persists_across_reopen(tmp_path):
    path = str(tmp_path / "g.sqlite")
    first = GuestBook(path, match_threshold=0.5)
    gid = first.assign(_unit(7))
    reopened = GuestBook(path, match_threshold=0.5)
    assert reopened.assign(_unit(7) + 0.02 * _unit(3)) == gid   # same stranger, later


def test_max_guests_caps_the_book(book):
    for s in range(6):               # max_guests=3
        book.assign(_unit(100 + s))
    assert book.count() <= 3


def test_guest_label():
    assert guest_label(-2) == "guest-2"
