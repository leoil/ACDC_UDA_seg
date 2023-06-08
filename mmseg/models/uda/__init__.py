# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add MinEnt and AdvSeg
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.advseg import AdvSeg
from mmseg.models.uda.dacs import DACS
from mmseg.models.uda.minent import MinEnt
from mmseg.models.uda.ckd import CKD
from mmseg.models.uda.ckd_single import CKD_single

__all__ = ['DACS', 'MinEnt', 'AdvSeg' , 'CKD', 'CKD_single']
