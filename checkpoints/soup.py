import math
import os
import random
from copy import deepcopy
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log

from tqdm import tqdm

model_pth1 = 'iter_32000.pth' # 7048
model_pth2 = 'iter_24000.pth' # 7035
model_pth3 = 'iter_40000.pth' # 7039


model1 = torch.load(model_pth1)
model2 = torch.load(model_pth2)
model3 = torch.load(model_pth3)

model_soup = {
    "state_dict":dict(),
    "meta": dict()
}

model_soup['meta']['PALETTE']= [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
model_soup['meta']['CLASSES']  = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

for (key, p1), (_, p2), (_, p3) in zip( model1['state_dict'].items(), model2['state_dict'].items(), model3['state_dict'].items()):
    if "imnet" not in key:
        if "ema" not in key:
            if "ca_"not in key:    
                # model_soup['state_dict'][key] = (p1 + p2 + p3)/3.0
                model_soup['state_dict'][key] = 0.334*p1 + 0.333*p2 + 0.333*p3



torch.save(model_soup, 'soup_fan_334.pth')# 7060

# fan_334
# +---------------+-------+-------+
# |     Class     |  IoU  |  Acc  |
# +---------------+-------+-------+
# |      road     | 90.98 | 98.48 |
# |    sidewalk   |  67.6 |  74.3 |
# |    building   | 87.54 | 92.17 |
# |      wall     | 60.35 | 73.99 |
# |     fence     | 47.31 | 51.81 |
# |      pole     | 63.23 | 68.95 |
# | traffic light | 74.95 | 90.12 |
# |  traffic sign | 67.89 | 84.41 |
# |   vegetation  | 74.51 | 94.85 |
# |    terrain    | 45.51 | 59.15 |
# |      sky      |  82.9 | 84.53 |
# |     person    | 69.87 |  76.5 |
# |     rider     | 44.15 | 56.04 |
# |      car      | 89.78 | 96.23 |
# |     truck     | 85.61 | 93.09 |
# |      bus      |  94.4 | 96.22 |
# |     train     | 92.29 |  95.6 |
# |   motorcycle  | 48.19 | 53.95 |
# |    bicycle    | 54.27 |  69.2 |
# +---------------+-------+-------+
# Summary:

# +-------+------+-------+
# |  aAcc | mIoU |  mAcc |
# +-------+------+-------+
# | 89.13 | 70.6 | 79.45 |

