# Dual Variational Generation for Low Shot HFR
A PyTorch code of paper [Dual Variational Generation for Low Shot Heterogeneous Face Recognition](https://arxiv.org/pdf/1903.10203.pdf).

## Our Heterogeneous Face Recognition Works

**DVG-Face: Dual Variational Generation for Heterogeneous Face Recognition** 

Chaoyou Fu, Xiang Wu, Yibo Hu, Huaibo Huang, Ran He. IEEE TPAMI 2021

**Dual Variational Generation for Low Shot Heterogeneous Face Recognition** 

Chaoyou Fu, Xiang Wu, Yibo Hu, Huaibo Huang, Ran He. NeurIPS 2019

**Towards Lightweight Pixel-Wise Hallucination for Heterogeneous Face Recognition** 

Chaoyou Fu, Xiaoqiang Zhou, Weizan He, Ran He. IEEE TPAMI 2022

**Cross-Spectral Face Hallucination via Disentangling Independent Factors** 

Boyan Duan, Chaoyou Fu, Yi Li, Xingguang Song, Ran He. CVPR 2020


## News
The extension version of DVG is published in IEEE TPAMI 2021 ([DVG-Face: Dual Variational Generation for Heterogeneous Face Recognition](https://arxiv.org/pdf/2009.09399.pdf)), and its code is released in https://github.com/BradyFU/DVG-Face.
The newly released extension version has more powerful performances than this version.

## Prerequisites
- Python 2.7
- Pytorch 0.4.1 & torchvision 0.2.1 

## Train the generator
- Download LightCNN-29 model ([Google Drive](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view)) pretrained on the MS-Celeb-1M dataset.
- Train the generator:
```
sh run_train_generator.sh
```
- Note that this is a simplified version of our original code: <br>
        1. The diversity loss and the adversarial loss in the paper are removed. <br>
        2. The distribution alignment loss is replaced by a Maximum Mean Discrepancy (MMD) loss.
- The generated results during training will be saved in `./results`.

## Generate images from noise
- Use the trained generator to sample 100,000 paired heterogeneous data:
```
Python val.py --pre_model './model/netG_model_epoch_50_iter_0.pth'
```
- The generated fake NIR and VIS images will be saved in `./fake_images/nir_noise` and `./fake_images/vis_noise`, respectively.

## Train the recognition model LightCNN-29
- Use the real data and the generated fake data to train lightcnn:
```
sh run_train_lightcnn.sh
```

## Performance
The performance on the 1-fold of CASIA NIR-VIS 2.0 dataset after running the above code:

Rank-1 | VR@FAR=0.1% | VR@FAR=0.01%
:---: | :---: | :---:
99.9% | 99.8% | 98.9%

## Citation
If you use our code for your research, please cite the following paper:
```
@article{fu2021dvg,
  title={DVG-face: Dual variational generation for heterogeneous face recognition},
  author={Fu, Chaoyou and Wu, Xiang and Hu, Yibo and Huang, Huaibo and He, Ran},
  journal={IEEE TPAMI},
  year={2021}
}

@inproceedings{fu2019dual,
  title={Dual Variational Generation for Low-Shot Heterogeneous Face Recognition},
  author={Fu, Chaoyou and Wu, Xiang and Hu, Yibo and Huang, Huaibo and He, Ran},
  booktitle={NeurIPS},
  year={2019}
}

@article{fu2022towards,
  title={Towards Lightweight Pixel-Wise Hallucination for Heterogeneous Face Recognition},
  author={Fu, Chaoyou and Zhou, Xiaoqiang and He, Weizan and He, Ran},
  journal={IEEE TPAMI},
  year={2022}
}

@inproceedings{duan2020cross,
  title={Cross-spectral face hallucination via disentangling independent factors},
  author={Duan, Boyan and Fu, Chaoyou and Li, Yi and Song, Xingguang and He, Ran},
  booktitle={CVPR},
  year={2020}
}

```
