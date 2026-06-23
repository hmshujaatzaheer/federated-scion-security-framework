"""Closed-loop controller: observe -> learn -> decide -> reconfigure."""

from .closed_loop import ClosedLoopController, LoopBudget

__all__ = ["ClosedLoopController", "LoopBudget"]
