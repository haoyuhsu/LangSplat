#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from preprocess import OpenCLIPNetwork, OpenCLIPNetworkConfig
from autoencoder.model import Autoencoder


def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args, clip_model, ae_model, relevancy_threshold):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy")
    gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    ############################
    clip_model.pos_embeds = clip_model.pos_embeds.to(torch.float32)
    clip_model.neg_embeds = clip_model.neg_embeds.to(torch.float32)
    clip_model.pos_embeds = ae_model.encode(clip_model.pos_embeds)
    clip_model.neg_embeds = ae_model.encode(clip_model.neg_embeds)
    ############################

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, args)

        if not args.include_feature:
            rendering = output["render"]
        else:
            rendering = output["language_feature_image"]
            
        if not args.include_feature:
            gt = view.original_image[0:3, :, :]
            
        else:
            gt, mask = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=args.feature_level)

        ############################
        # compute the relevancy map for the current view (use encoding)
        rendering = rendering.to("cuda:0")
        rendering = rendering.permute(1,2,0)  # H x W x C
        h, w, c = rendering.shape
        rendering = rendering.reshape(-1, c)
        relevancy_maps = clip_model.get_relevancy(rendering[..., 0:3], 0)

        relevancy_maps = relevancy_maps[..., 0]  # (H x W)

        relevancy_threshold = float(args.rel_threshold)

        # smooth the relevancy map by applying a mean filter with 17 x 17 kernel
        relevancy_maps = relevancy_maps.reshape(h, w)    # reshape back to H x W x C
        relevancy_maps = relevancy_maps.unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W
        relevancy_maps = torch.nn.functional.avg_pool2d(relevancy_maps, kernel_size=(17, 17), stride=1, padding=8)  # 1 x 1 x H x W
        relevancy_maps = relevancy_maps.squeeze(0).squeeze(0)  # H x W
        relevancy_maps = relevancy_maps.reshape(-1)  # (H x W)

        query_mask = relevancy_maps > relevancy_threshold

        # assert torch.sum(query_mask) > 0, "No relevant features found for the query"

        # set colors & alphas in rendering to 0 for non-mask regions
        rendering[torch.logical_not(query_mask), :] = 0
        rendering = rendering.reshape(h, w, c)    # reshape back to H x W x C
        rendering = rendering.permute(2,0,1)  # C x H x W
        ############################

        np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        np.save(os.path.join(gts_npy_path, '{0:05d}'.format(idx) + ".npy"),gt.permute(1,2,0).cpu().numpy())
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
               
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')

        # initiliaze OpenCLIP model
        text = args.text_query
        config = OpenCLIPNetworkConfig()
        # for attr, value in OpenCLIPNetworkConfig.__dict__.items():
        #     print(f"{attr}: {value}")
        clip_model = OpenCLIPNetwork(config)
        clip_model.set_positives([text])

        # initiliaze Autoencoder model
        ae_ckpt_path = f"ckpt/teatime/ae_ckpt/best_ckpt.pth"
        ae_checkpoint = torch.load(ae_ckpt_path)
        ae_model = Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512]).to("cuda:0")
        ae_model.load_state_dict(ae_checkpoint)
        ae_model.eval()

        gaussians_encoded_features = gaussians._language_feature.to('cuda:0')

        # Option 1: decode the encoded features (from dim-3 to dim-512)
        # gaussians_clip_features = ae_model.decode(gaussians_encoded_features)
        # relevancy_maps = clip_model.get_relevancy(gaussians_clip_features, 0)

        # Option 2: encode the clip model embeddings (from dim-512 to dim-3)
        # clip_model.pos_embeds = clip_model.pos_embeds.to(torch.float32)
        # clip_model.neg_embeds = clip_model.neg_embeds.to(torch.float32)
        # clip_model.pos_embeds = ae_model.encode(clip_model.pos_embeds)
        # clip_model.neg_embeds = ae_model.encode(clip_model.neg_embeds)
        # relevancy_maps = clip_model.get_relevancy(gaussians_encoded_features, 0)

        relevancy_threshold = float(args.rel_threshold)
        # query_mask = relevancy_maps[..., 0] > relevancy_threshold

        # assert torch.sum(query_mask) > 0, "No relevant features found for the query"

        # Updatable parameters of 3D gaussians
        #     self._xyz, 
        #     self._features_dc, 
        #     self._features_rest,
        #     self._scaling, 
        #     self._rotation, 
        #     self._opacity,
        #     self.max_radii2D,
        # gaussians._xyz = gaussians._xyz[query_mask]
        # gaussians._features_dc = gaussians._features_dc[query_mask]
        # gaussians._features_rest = gaussians._features_rest[query_mask]
        # gaussians._language_feature = gaussians._language_feature[query_mask]
        # gaussians._scaling = gaussians._scaling[query_mask]
        # gaussians._rotation = gaussians._rotation[query_mask]
        # gaussians._opacity = gaussians._opacity[query_mask]
        # gaussians.max_radii2D = gaussians.max_radii2D[query_mask]
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, dataset.source_path, "text_{}_{:.2f}".format(text, relevancy_threshold), scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args, \
                       clip_model, ae_model, relevancy_threshold)

        # if not skip_test:
        #     render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")

    parser.add_argument("--text_query", default="object", type=str)
    parser.add_argument("--rel_threshold", default=0.6, type=float)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)