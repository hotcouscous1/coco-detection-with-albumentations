import os
import math
from typing import List, Tuple, Dict, OrderedDict, Optional, Union

import cv2
import torch
import numpy as np

Numpy = np.array
Tensor = torch.Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')