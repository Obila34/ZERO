"""Privacy & consent — a module, not a footnote.

An always-listening robot with a camera, face embeddings and a memory of
people is exactly the system the ambient-agent literature warns about. The
stance here:

  * ON-DEVICE   — audio, frames, embeddings and memories never leave the
                  Pi/GPU pair; nothing is cloud-bound (see OPERATIONS.md).
  * CONSENT     — people enter the registry only by introducing THEMSELVES
                  ("I'm David"); ZERO never enrols someone who didn't ask.
  * BYSTANDERS  — configurable: strangers can be answered but never remembered
                  (guarded), or not engaged at all (strict).
  * ERASURE     — "forget that" and "forget everything about me" are honoured
                  immediately, out loud.
  * INDICATOR   — a visible listening signal (GPIO LED when wired, logs
                  otherwise) so being recorded is never a surprise.
"""
from zero.privacy.guard import PrivacyGuard, parse_forget_command
from zero.privacy.indicator import ListeningIndicator

__all__ = ["PrivacyGuard", "parse_forget_command", "ListeningIndicator"]
