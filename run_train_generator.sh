#!/bin/bash

echo train generator

gpu_ids='0'
workers=8
batch_size=8
all_epochs=50
pre_epoch=0
hdim=128
test_epoch=10
save_epoch=10

lambda_mmd=50
lambda_ip=1000
lambda_pair=5

ip_model='./LightCNN_29Layers_V2_checkpoint.pth.tar'
img_root='path of your dataset'
train_list='path of your training list'

python train_generator.py --gpu_ids $gpu_ids --workers $workers --batch_size $batch_size --all_epochs $all_epochs \
                          --pre_epoch $pre_epoch --hdim $hdim --test_epoch $test_epoch --save_epoch $save_epoch \
                          --lambda_mmd $lambda_mmd --lambda_ip $lambda_ip --lambda_pair $lambda_pair \
                          --ip_model $ip_model --img_root $img_root --train_list $train_list | tee train_generator.log
