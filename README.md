# Dual Variational Generation for Low Shot HFR
A [pytorch](https://pytorch.org/) code of paper [Dual Variational Generation for Low Shot Heterogeneous Face Recognition](https://arxiv.org/pdf/1903.10203.pdf), written with [Xiang Wu](https://github.com/AlfredXiangWu).

## Prerequisites
- Python 2.7
- Pytorch 0.4.1 && torchvision 0.2.1 

## Train the generator
- Download LightCNN-29 model ([Google Drive](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view)) pretrained on the MS-Celeb-1M dataset.
- Train the generator:
```
sh run_train_generator.sh
```
- Note that this is a simplified version of our original code: <br>
        1.  The diversity loss and the adversarial loss in the paper are removed. <br>
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
The performance on the CASIA NIR-VIS 2.0 dataset after running the above code:

Rank-1 | VR@FAR=0.1% | VR@FAR=0.01%
:---: | :---: | :---:
99.9% | 99.8% | 98.9%

## Citation
If you use our code for your research, please cite the following paper:
```
@inproceedings{fu2019dual,
  title={Dual Variational Generation for Low-Shot Heterogeneous Face Recognition},
  author={Fu, Chaoyou and Wu, Xiang and Hu, Yibo and Huang, Huaibo and He, Ran},
  booktitle={NeurIPS},
  year={2019}
}
```
