#!/bin/bash

dataset_path=./datasets/counter
casename=counter

# get the language feature of the scene
# python preprocess.py --dataset_path $dataset_path --sam_ckpt_path ./ckpts/sam_vit_h_4b8939.pth

# train the autoencoder
# cd autoencoder
# python autoencoder/train.py --dataset_path $dataset_path --dataset_name $casename --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --output ae_ckpt
# get the 3-dims language feature of the scene
# python autoencoder/test.py --dataset_path $dataset_path --dataset_name $casename --output ae_ckpt

# ATTENTION: Before you train the LangSplat, please follow https://github.com/graphdeco-inria/gaussian-splatting
# to train the RGB 3D Gaussian Splatting model.
# put the path of your RGB model after '--start_checkpoint'

# for level in 1 2 3
# do
#     python train.py -s $dataset_path -m output/${casename} --start_checkpoint $dataset_path/output/$casename/chkpnt30000.pth --feature_level ${level}
# done

for level in 1 2 3
do
    # render rgb
    python render.py -m output/${casename}_${level}
    # render language features
    python render.py -m output/${casename}_${level} --include_feature
done


dataset_path=./datasets/teatime
casename=teatime

# get the language feature of the scene
# python preprocess.py --dataset_path $dataset_path --sam_ckpt_path ./ckpts/sam_vit_h_4b8939.pth

# train the autoencoder
# cd autoencoder
# python autoencoder/train.py --dataset_path $dataset_path --dataset_name $casename --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --output ae_ckpt
# get the 3-dims language feature of the scene
# python autoencoder/test.py --dataset_path $dataset_path --dataset_name $casename --output ae_ckpt

# ATTENTION: Before you train the LangSplat, please follow https://github.com/graphdeco-inria/gaussian-splatting
# to train the RGB 3D Gaussian Splatting model.
# put the path of your RGB model after '--start_checkpoint'

# for level in 1 2 3
# do
#     python train.py -s $dataset_path -m output/${casename} --start_checkpoint $dataset_path/output/$casename/chkpnt30000.pth --feature_level ${level}
# done

for level in 1 2 3
do
    # render rgb
    python render.py -m output/${casename}_${level}
    # render language features
    python render.py -m output/${casename}_${level} --include_feature
done