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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

from utils.general_utils import PILtoTorch
from PIL import Image

from functools import lru_cache

class Camera(nn.Module):

    ########## This does not work ############
    # @staticmethod
    # @lru_cache(maxsize=100)  # Adjust maxsize as needed
    # def _load_and_process_image(image_path, resolution, device):
    #     image_pil = Image.open(image_path)
    #     image_torch = PILtoTorch(image_pil, resolution).to(device)

    #     image = image_torch[:3, ...]
    #     gt_alpha_mask = None
    #     if image_torch.shape[0] == 4:
    #         gt_alpha_mask = image_torch[3:4, ...]

    #     original_image = image.clamp(0.0, 1.0)
    #     image_width = original_image.shape[2]
    #     image_height = original_image.shape[1]

    #     if gt_alpha_mask is not None:
    #         original_image *= gt_alpha_mask.to(device)
    #     else:
    #         original_image *= torch.ones((1, image_height, image_width), device=device)

    #     return original_image

    @property
    def original_image(self):

        image_pil = Image.open(self.image_path)
        image_torch = PILtoTorch(image_pil, (self.image_width, self.image_height))

        image = image_torch[:3, ...]
        gt_alpha_mask = None
        if image_torch.shape[1] == 4:
            gt_alpha_mask = image_torch[3:4, ...]

        original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = original_image.shape[2]
        self.image_height = original_image.shape[1]

        if gt_alpha_mask is not None:
            original_image *= gt_alpha_mask.to(self.data_device)
        else:
            original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        # original_image = Camera._load_and_process_image(self.image_path, (self.width, self.height), self.data_device)
        return original_image

    def __init__(self, colmap_id, R, T, FoVx, FoVy, image_name, uid, image_path, depth_path, resolution,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.image_path = image_path
        self.depth_path = depth_path
        self.image_width, self.image_height = resolution

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")


        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

