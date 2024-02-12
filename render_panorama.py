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

# from utils.graphics_utils import getProjectionMatrix
from scene.cameras import Camera
from utils.py360_utils import c2e
from PIL import Image
import math

               
def render_panorama(dataset : ModelParams, pipeline : PipelineParams, views, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        # scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(dataset.model_path, "panorama")
        makedirs(render_path, exist_ok=True)

        cube_map_dict = {}
        for view_name, view in tqdm(views.items(), desc="Rendering progress"):
            output = render(view, gaussians, pipeline, background, args)
            rendering = output["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, view_name + ".png"))
            cube_map_dict[view_name] = rendering.permute(1,2,0).cpu().numpy()

        # convert cubemap to equirectangular
        h, w = 1024, 2048
        equirectangular = c2e(cube_map_dict, h, w, mode='bilinear', cube_format='dict')
        equirectangular = Image.fromarray(np.clip(equirectangular * 255, 0, 255).astype(np.uint8))
        equirectangular.save(os.path.join(render_path, "equirectangular.png"))


if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    # sample views into 6 directions to form a skybox cubemap
    # front: +X, back: -X, left: +Y, right: -Y, up: +Z, down: -Z
    Z_FAR = 100.0
    Z_NEAR = 0.01
    FOV = math.pi / 2
    ASPECT_RATIO = 1.0
    IMG_SIZE = 1024
    center = np.array([0, 0, 0])
    image = torch.zeros(3, IMG_SIZE, IMG_SIZE)

    def create_camera(view_name, colmap_id, lookat, scene_up, center, FOV, image=None, gt_alpha_mask=None, data_device="cuda"):
        right = np.cross(lookat, scene_up)
        up = np.cross(right, lookat)
        R = np.array([right, -up, lookat]).T  # OPENCV
        T = -center
        return Camera(colmap_id=colmap_id, R=R, T=T, FoVx=FOV, FoVy=FOV, image=image, gt_alpha_mask=gt_alpha_mask, image_name=view_name, uid=colmap_id, data_device=data_device)

    views = {
        "front": create_camera("front", 0, np.array([1, 0, 0]),  np.array([0, 0, 1]),  center, FOV, image),
        "back" : create_camera("back",  1, np.array([-1, 0, 0]), np.array([0, 0, 1]),  center, FOV, image),
        "left" : create_camera("left",  2, np.array([0, 1, 0]),  np.array([0, 0, 1]),  center, FOV, image),
        "right": create_camera("right", 3, np.array([0, -1, 0]), np.array([0, 0, 1]),  center, FOV, image),
        "up"   : create_camera("up",    4, np.array([0, 0, 1]),  np.array([-1, 0, 0]), center, FOV, image),
        "down" : create_camera("down",  5, np.array([0, 0, -1]), np.array([1, 0, 0]),  center, FOV, image),
    }

    render_panorama(model.extract(args), pipeline.extract(args), views, args)