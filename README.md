## Short Description：

* Baseline model： MIC with FAN pretrained backbone([NVlabs/FAN: Fully Attentional Networks](https://github.com/NVlabs/FAN#fan-hybrid-imagenet-22k-trained-models))

* Use the MIC checkpoint at ([MIC/seg · lhoyer/MIC](https://github.com/lhoyer/MIC/tree/master/seg#checkpoints)) with MiT-B5 backbone as  pseudo-label teacher teacher model, and use it for knowledge distillation.

* Model Soup: Weighted-sum for ditilled FAN model (3 model in total-24k, 32k, 40k), the seperate models is attached below.

## Environment Setup & Data Preparation

Mostly following  [MIC](https://github.com/lhoyer/MIC/tree/master/seg#environment-setup) , with few modification:

```shell
timm==0.5.4
mmcv-full==1.7.0
torch==1.10.0+cu111 
torchvision==0.11.0+cu111 
python==3.8.15
```
Donwload the pretrained FAN-Large Backbone via [FAN-large](https://github.com/zhoudaquan/fully_attentional_network_ckpt/releases/download/v1.0.0/fan_hybrid_large_in22k_1k.pth.tar), and link it in the `pretrained` dir: 

```shell
ln -s fan_hybrid_large_in22k_1k.pth.tar  fan_large_16_p4_hybrid.pth.tar
```

## Hardware information

V100_32G x1


## Knowledge distillation

Knowledge distillation is done by running:

```shell
python run_expriments.py --config config/mic/csHR2acdcHR_mic_ckd_hrda_fan_single.py 
```

Here, the MIC-HRDA Model is used as the teacher model, and the FAN model is used as the student model, `fd-things dist` is not implemented.

## Model Soup

place fan_iter_24k.pth, fan_iter_32k.pth, fan_iter_40k.pth in the `./checkpoints` dir, then run:

```shell
python soup.py
```

## Inference:
Inference on test dataset is done by running:

```shell
python -m tools.test configs/mic/cs2acdc_mic_ckd_fan.py   checkpoints/soup_fan_334.pth \
--test-set --format-only --eval-option imgfile_prefix=labelTrainIds  to_label_id=False
```

## Checkpoints for quick reproducibility check:

Checkpoint of final model is available at [Cityscapes2ACDC_UDA_USTC-IAT.zip](https://drive.google.com/file/d/1DjnEElinIAapAU1dpCy80o_LDmYCDLL4/view?usp=sharing)

```
soup_fan_334.pth:                                               Model weight after model soup.
cs2acdc_mic_ckd_fan.py:                                         config file for inferencing
cs2acdc_mic_kd_fan_seed29                                       work dir folder of the Knowledge Distillation of FAN model.
    ├── 20230602_123857.log:                                    log file
    ├── 20230602_123857.log.json
    ├── 230602_1238_csHR2acdcHR_mic_hrda_s2_8f896.py:           config file
    ├── code.tar.gz:                                            snapshot of source code
```


## Checkpoint Download:

[fan_iter_24k.pth](https://drive.google.com/file/d/13ZNhD21gl4AcqnY1wPVRCZeOcTn0D_f2/view?usp=sharing)

[fan_iter_32k.pth](https://drive.google.com/file/d/1yP_JPTDHPeuPU_QguedZ9vabYpS1iJdn/view?usp=drive_link)

[fan_iter_40k.pth](https://drive.google.com/file/d/1Q7h9cQlsuuFFkm2ObvXyGxS9k7YG8vUq/view?usp=sharing)

[soup_fan_334.pth](https://drive.google.com/file/d/1PhOclIkcmn4GHJkUqU_FTkVM7L22hezA/view?usp=drive_link)

[Knowledge Distillation Log](https://drive.google.com/file/d/1mlWZAlt7R6Oi0wC-k8XvFfSo_lfCin1d/view?usp=drive_link)

<!-- Name：Quansheng Liu

Organization： University of Science and Technology of China-IAT -->


## Acknowledgements

* [MMsegmentation](https://github.com/open-mmlab/mmsegmentation)
* [MIC](https://github.com/lhoyer/MIC)
* [FAN](https://github.com/NVlabs/FAN)