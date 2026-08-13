"""ZERO's head / attention subsystem.

Revived from hub/main@28ce9ee (it was never reverted for cause — it was built on
a parallel branch this one forked before, and never merged). The pure-logic
motion core (HeadController) and closed-loop visual servo (FaceTracker) are
brought forward intact; the transport and composition layers are rebuilt against
this branch's current vision/identity APIs.

Import policy: this package's top level exposes ONLY the dependency-light pieces
(pure logic + the driver layer), so `import zero.head` never drags in cv2, the
identity stack, or sockets. Heavier composition (HeadSystem) is imported lazily
by the factory when head.enabled is true.
"""
from __future__ import annotations

from zero.head.controller import HeadController
from zero.head.tracker import FaceTracker
from zero.head.driver import NullDriver, make_driver

__all__ = ["HeadController", "FaceTracker", "NullDriver", "make_driver"]
