# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes High-Resolution Data Loading
    '../_base_/datasets/uda_cityscapesHR_to_acdcHR_1024x1024.py',
    # DAFormer Self-Training
    '../_base_/uda/dacs_a999_fdthings.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'

]

ema_cfg = dict(
    type='HRDAEncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='mit_b5', style='pytorch'),
    decode_head=dict(
        type='HRDAHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=dict(type='BN', requires_grad=True))),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        single_scale_head='DAFormerHead',
        attention_classwise=True,
        hr_loss_weight=0.1),
    train_cfg=dict(
        log_config=dict(
            interval=50,
            img_interval=1000,
            hooks=[dict(type='TextLoggerHook', by_epoch=False)])
    ),
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[512, 512],
        crop_size=[1024, 1024]),
    scales=[1, 0.5],
    hr_crop_size=(512, 512),
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True)




# Random Seed
seed = 29  # seed with median performance
# HRDA Configuration
model = dict(
    type='HRDAEncoderDecoder',
    pretrained='pretrained/fan_large_16_p4_hybrid.pth.tar',    
    backbone=dict(type='fan_large_16_p4_hybrid', style='pytorch'),    
    decode_head=dict(
        in_channels=[128, 256, 480, 480],
        
        type='HRDAHead',
        # Use the DAFormer decoder for each scale.
        single_scale_head='DAFormerHead',
        # Learn a scale attention for each class channel of the prediction.
        attention_classwise=True,
        # Set the detail loss weight $\lambda_d=0.1$.
        hr_loss_weight=0.1,
        
        loss_decode=dict(
            type='LogitConstraintLoss', use_sigmoid=False, loss_weight=1.0)        
        
    ),
    # Use the full resolution for the detail crop and half the resolution for
    # the context crop.
    scales=[1, 0.5],
    # Use a relative crop size of 0.5 (=512/1024) for the detail crop.
    hr_crop_size=(512, 512),
    # Use LR features for the Feature Distance as in the original DAFormer.
    feature_scale=0.5,
    # Make the crop coordinates divisible by 8 (output stride = 4,
    # downscale factor = 2) to ensure alignment during fusion.
    crop_coord_divisible=8,
    # Use overlapping slide inference for detail crops for pseudo-labels.
    hr_slide_inference=True,
    # Use overlapping slide inference for fused crops during test time.
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[512, 512],
        crop_size=[1024, 1024]),
    


)
data = dict(
    train=dict(
        # Rare Class Sampling
        # min_crop_ratio=2.0 for HRDA instead of min_crop_ratio=0.5 for
        # DAFormer as HRDA is trained with twice the input resolution, which
        # means that the inputs have 4 times more pixels.
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=2.0),
        # Pseudo-Label Cropping v2 (from HRDA):
        # Generate mask of the pseudo-label margins in the data loader before
        # the image itself is cropped to ensure that the pseudo-label margins
        # are only masked out if the training crop is at the periphery of the
        # image.
        # target=dict(crop_pseudo_margins=[30, 240, 30, 30]),
    ),
    # Use one separate thread/worker for data loading.
    workers_per_gpu=1,
    # Batch size
    samples_per_gpu=2,
)
# MIC Parameters
uda = dict(
    ema_cfg = ema_cfg,
    
    type = 'CKD_single',
    # mic_ori_70
    ca_model_load_from="/home/liuqs/workspace/code/MIC/seg/pretrained/csHR2acdcHR_mic_hrda_b52f1/latest.pth",
    # 
    # cu_model_load_from="/home/liuqs/workspace/code/MIC/seg/work_dirs/local-exp80/230530_0106_csHR2acdcHR_1024x1024_dacs_a999_fdthings_rcs001-20_m64-07-sep_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s1_c6dbe/latest.pth",


    ca_model_cfg = ema_cfg,
    # cu_model_cfg = ema_cfg,    
    
    
    enable_vb = True, 
    # enable_vb = False,
    # enable_fdist = False,
    enable_fdist = True, 
    
    # Apply masking to color-augmented target images
    # mask_mode='separatetrgaug',
    mask_mode='separate',    
    # Use the same teacher alpha for MIC as for DAFormer
    # self-training (0.999)
    mask_alpha='same',
    # Use the same pseudo label confidence threshold for
    # MIC as for DAFormer self-training (0.968)
    mask_pseudo_threshold='same',
    # Equal weighting of MIC loss
    mask_lambda=1,
    # Use random patch masking with a patch size of 64x64
    # and a mask ratio of 0.7
    mask_generator=dict(
        type='block', mask_ratio=0.7, mask_block_size=64, _delete_=True))
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
gpu_model = 'NVIDIATITANRTX'
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=4)
# evaluation = dict(interval=4000, metric='mIoU')
evaluation = dict(interval=4000, metric='mIoU')
# Meta Information for Result Analysis
name = 'csHR2acdcHR_mic_hrda_s2'
exp = 'basic'
name_dataset = 'cityscapesHR2acdcHR_1024x1024'
name_architecture = 'hrda1-512-0.1_daformer_sepaspp_sl_mitb5'
name_encoder = 'mitb5'
name_decoder = 'hrda1-512-0.1_daformer_sepaspp_sl'
name_uda = 'dacs_a999_fdthings_rcs0.01-2.0_cpl2_m64-0.7-spta'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'

# For the other configurations used in the paper, please refer to experiment.py
