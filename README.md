# MMCosine_ICASSP23
 This is the code release for ICASSP 2023 Paper "*MMCosine: Multi-Modal Cosine Loss Towards Balanced Audio-Visual Fine-Grained Learning*", implemented with Pytorch.

**Title:** **MMCosine: Multi-Modal Cosine Loss Towards Balanced Audio-Visual Fine-Grained Learning**

**Authors: [Ruize Xu](https://rick-xu315.github.io/), Ruoxuan Feng, Shi-xiong Zhang, [Di Hu](https://dtaoo.github.io/)**

:rocket: Project page here: [Project Page](https://gewu-lab.github.io/MMCosine/)

:page_facing_up: Paper here: [Paper](https://arxiv.org/abs/2303.05338)

:mag: Supplementary material: [Supplementary](https://rick-xu315.github.io/ICASSP23_Sup.pdf)



## Overview

Recent studies show that the imbalanced optimization of uni-modal encoders in a joint-learning model is a bottleneck to enhancing the model`s performance. We further find that the up-to-date imbalance-mitigating methods fail on some audio-visual fine-grained tasks, which have a higher demand for distinguishable feature distribution. Fueled by the success of cosine loss that builds hyperspherical feature spaces and achieves lower intra-class angular variability, this paper proposes Multi-Modal Cosine loss, *MMCosine*. It performs a modality-wise $L_2$ normalization to features and weights towards balanced and better multi-modal fine-grained learning.

## Data Preparation

- Download Original Dataset: [CREMAD](https://github.com/CheyneyComputerScience/CREMA-D), [SSW60](https://github.com/visipedia/ssw60), [Voxceleb1&2](https://mm.kaist.ac.kr/datasets/voxceleb/), and [UCF 101(supplementary)](https://www.crcv.ucf.edu/research/data-sets/ucf101/).

- Preprocessing:
  - CREMAD: Refer to [OGM-GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022) for video processing.
  - SSW60: Refer to the [original repo](https://github.com/visipedia/ssw60) for details.
  - Voxceleb1&2: After extracting frames (2fps) from the raw video, we utilize [RetinaFace](https://github.com/serengil/retinaface) to extract and align faces. The official pipeline trains on Voxceleb2 and test on the Voxceleb1 test set, and we add validation on the manually-made Voxceleb2 test set. The annotation is in ```/data``` folder.

## Main Dependencies

- ubuntu 18.04
- CUDA Version: 11.6
- Python: 3.9.7
- torch: 1.10.1
- torchaudio: 0.10.1 
- torchvision: 0.11.2 

## Run

 You can train your model on the provided datasets (*e.g.* CREMAD) simply by running:

``` python main_CD.py --train --fusion_method gated --mmcosine True --scaling 10```

Apart from fusion methods and scaling parameters, you can also adjust the setting such as ```batch_size```, ```lr_decay```, ```epochs```, *etc*.

You can also record intermediate variables through [tensorboard](https://pytorch.org/docs/stable/tensorboard.html) by nominating ```use_tensorboard``` and ```tensorboard_path``` for saving logs.

## Bibtex

If you find this work useful, please consider citing it.

```BibTeX
@inproceedings{xu2023mmcosine,
  title={MMCosine: Multi-Modal Cosine Loss Towards Balanced Audio-Visual Fine-Grained Learning},
  author={Xu, Ruize and Feng, Ruoxuan and Zhang, Shi-Xiong and Hu, Di},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgement

This research was supported by Public Computing Cloud, Renmin University of China.

## Contact us

If you have any detailed questions or suggestions, you can email us: <xrz0315@ruc.edu.cn>
