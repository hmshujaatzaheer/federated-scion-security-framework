"""
Signet-style Verifiable Notification (coordination stub)
========================================================

The privacy-preserving bandwidth market must settle cross-AS without trusting a
central sequencer. We anchor settlement to a
*verifiable proof of notification*, conceptually after:

    E. Ehsani Moghadam, M. Wyss, J. Kwon, M. Frei, Y.-C. Hu, A. Perrig, A. Sonnino,
    "Signet: Scalable Network-Driven Proof of Notification for Blockchain Systems,"
    IEEE ICDCS 2026.

This is an *interface + reference stub*, not a reimplementation of Signet. It
provides a hash-chained, commitment-based notification record that a smart
contract can check, so the ZK market layer has a concrete coordination point to
build on. Cryptography here is illustrative (SHA-256 commitments); production
use would substitute the real protocol's network-driven proofs.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional


def _h(*parts: bytes) -> str:
    d = hashlib.sha256()
    for p in parts:
        d.update(p)
    return d.hexdigest()


@dataclass(frozen=True)
class Notification:
    """A single network-driven notification of a market event."""
    epoch: int
    payload_commitment: str   # e.g. hash of an (encrypted) reservation delivery
    prev_hash: str
    timestamp: float = field(default_factory=time.time)

    def digest(self) -> str:
        body = json.dumps(
            {"epoch": self.epoch, "c": self.payload_commitment, "p": self.prev_hash},
            sort_keys=True,
        ).encode()
        return _h(body)


class SignetLog:
    """
    Append-only, hash-chained notification log.

    Any party can verify that a notification was included and ordered, without a
    trusted sequencer: the chain of `prev_hash` links binds order, and the
    commitment hides the (private) payload while proving it was notified.
    """

    GENESIS = "0" * 64

    def __init__(self) -> None:
        self._entries: List[Notification] = []

    @property
    def head(self) -> str:
        return self._entries[-1].digest() if self._entries else self.GENESIS

    def notify(self, payload_commitment: str) -> Notification:
        n = Notification(
            epoch=len(self._entries),
            payload_commitment=payload_commitment,
            prev_hash=self.head,
        )
        self._entries.append(n)
        return n

    def verify_chain(self) -> bool:
        """Verify hash-chain integrity end to end."""
        prev = self.GENESIS
        for n in self._entries:
            if n.prev_hash != prev:
                return False
            prev = n.digest()
        return True

    def proof_of_notification(self, epoch: int) -> Optional[dict]:
        """Minimal inclusion proof a verifier (or smart contract) can check."""
        if not (0 <= epoch < len(self._entries)):
            return None
        n = self._entries[epoch]
        return {
            "epoch": n.epoch,
            "payload_commitment": n.payload_commitment,
            "prev_hash": n.prev_hash,
            "digest": n.digest(),
        }


def _demo() -> None:
    print("Signet-style Verifiable Notification (stub)")
    print("=" * 60)
    log = SignetLog()
    # A buyer's encrypted bandwidth-reservation delivery is committed and notified.
    for i in range(3):
        commit = _h(f"encrypted-reservation-{i}".encode())
        n = log.notify(commit)
        print(f"  notified epoch {n.epoch}: digest {n.digest()[:16]}...")
    print(f"  chain integrity verified: {log.verify_chain()}")
    print(f"  inclusion proof @epoch 1: {log.proof_of_notification(1)['digest'][:16]}...")


if __name__ == "__main__":
    _demo()
