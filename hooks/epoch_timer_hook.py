"""Hook that logs wall-clock training time per epoch."""

from __future__ import annotations

import time
from typing import Optional

from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine.registry import HOOKS


@HOOKS.register_module()
class EpochTimerHook(Hook):
    """Log wall-clock duration of each training epoch.

    Emits an ``INFO`` log line at the end of every training epoch and
    writes ``train/epoch_time_sec`` to the message hub so it surfaces
    in the WandB backend alongside the standard scalars.
    """

    priority = 'LOW'

    def __init__(self) -> None:
        self._t0: Optional[float] = None

    def before_train_epoch(self, runner) -> None:  # noqa: D401
        self._t0 = time.perf_counter()

    def after_train_epoch(self, runner) -> None:  # noqa: D401
        if self._t0 is None:
            return
        elapsed = time.perf_counter() - self._t0
        self._t0 = None

        logger = MMLogger.get_current_instance()
        logger.info(
            f'[EpochTimer] epoch {runner.epoch + 1}/{runner.max_epochs} '
            f'took {elapsed:.1f}s'
        )
        runner.message_hub.update_scalar('train/epoch_time_sec', elapsed)
