#!/bin/bash

echo extract features

gpu_ids='0'
batch_size=32
save_path='./feat'
weights='./model/lightCNN_model_epoch_12_iter_0.pth'
root_path='/data1/chaoyou.fu/HFR_Datasets/CASIA_NIR_VIS_align3'

img_list='/data1/chaoyou.fu/HFR_Datasets/CASIA_NIR_VIS_align3/list_file/NIR_VIS_probe1.txt'
python extract_features.py --gpu_ids $gpu_ids --batch_size $batch_size --save_path $save_path --weights $weights \
                           --root_path $root_path --img_list $img_list


img_list='/data1/chaoyou.fu/HFR_Datasets/CASIA_NIR_VIS_align3/list_file/NIR_VIS_gallery1.txt'
python extract_features.py --gpu_ids $gpu_ids --batch_size $batch_size --save_path $save_path --weights $weights \
                           --root_path $root_path --img_list $img_list
