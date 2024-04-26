# Author: Sun LuoHao
# All rights reserved
import lightning as L
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Mapping, Iterable
from torch import Tensor
from abc import ABC, abstractmethod
from ..framework_base.framework_base import FrameworkBase


class RegressionFramework(FrameworkBase, ABC): ...
