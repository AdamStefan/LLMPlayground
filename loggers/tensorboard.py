import logging
import os
from argparse import Namespace
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.tensorboard import SummaryWriter


log = logging.getLogger(__name__)


class TensorBoardLogger(object):
    def __init__(
        self,
        save_dir: str,
        **kwargs,
    ):
        super().__init__()
        self.save_dir = save_dir
        self._kwargs = kwargs

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.logger = SummaryWriter(log_dir=self.save_dir, **self._kwargs)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if int(os.environ.get("RANK", 0)) != 0:
            return None

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                self.logger.add_scalars(k, v, step)
            else:
                try:
                    self.logger.add_scalar(k, v, step)
                # todo: specify the possible exception
                except Exception as ex:
                    m = f"\n you tried to log {v} which is not currently supported. Try a dict or a scalar/tensor."
                    raise ValueError(m) from ex

    def finalize(self) -> None:
        self.logger.flush()
        self.logger.close()

    @property
    def name(self) -> str:
        return self._name
