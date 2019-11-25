# Dual Variational Generation for Low Shot HFR
A [pytorch](https://pytorch.org/) implementation of paper [Dual Variational Generation for Low Shot Heterogeneous Face Recognition](https://arxiv.org/pdf/1903.10203.pdf)

## Prerequisites
- Python 2.7
- Pytorch 0.4.1 && torchvision 0.2.1 

## Training
- Download the pretrained LightCNN-29 model on [Google Drive](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view), and put it in the `pre_train` folder.
- Train the generation part: <br>
`sh run_train_generator.sh`
