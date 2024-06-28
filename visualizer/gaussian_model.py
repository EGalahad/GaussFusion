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
import numpy as np
import math
from typing import NamedTuple
from torch import nn
import os
from plyfile import PlyData, PlyElement
from sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from render import render, Camera


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions()

        self.bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device="cuda")

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    @torch.inference_mode()
    def render(
        self, w2c, intrinsics, bg_color=None, scaling_modifier=1.0, override_color=None
    ):
        def create_camera(w2c, intrinsics):
            # Create world_view_transform
            #    [R^T, 0],
            #    [t, 1]
            world_view_transform = torch.as_tensor(w2c).t()
            rot = world_view_transform[:3, :3].t()
            tran = world_view_transform[3, :3]

            width = intrinsics["width"]
            height = intrinsics["height"]
            focal_x = intrinsics["focal_x"]
            focal_y = intrinsics["focal_y"]

            intrinsics_matrix = torch.eye(4)
            # intrinsics_matrix:
            #    [2 * fx / w, 0, 0, 0],
            #    [0, 2 * fy / h, 0, 0],
            #    [0, 0, 1, 1]
            #    [0, 0, 0, 0]
            intrinsics_matrix[0, 0] = 2 * focal_x / width
            intrinsics_matrix[1, 1] = 2 * focal_y / height
            intrinsics_matrix[2, 3] = 1
            intrinsics_matrix[3, 3] = 0
            full_proj_transform = world_view_transform @ intrinsics_matrix

            # Calculate FoVx and FoVy
            FoVx = 2 * math.atan(width / (2 * focal_x))
            FoVy = 2 * math.atan(height / (2 * focal_y))

            # Calculate camera_center
            camera_center = -torch.matmul(rot.t(), tran)

            # c2w = np.linalg.inv(w2c)
            # print(c2w[:3, 3], camera_center, tran)

            # Create and return the Camera object
            return Camera(
                FoVx=FoVx,
                FoVy=FoVy,
                image_height=height,
                image_width=width,
                world_view_transform=world_view_transform.cuda(),
                full_proj_transform=full_proj_transform.cuda(),
                camera_center=camera_center.cuda(),
            )

        cam = create_camera(w2c, intrinsics)
        # print(cam.FoVx, cam.FoVy)
        # print(cam.full_proj_transform)
        # print(cam.world_view_transform)
        # print(cam.camera_center)
        if bg_color is None:
            bg_color = self.bg_color
        return render(cam, self, None, bg_color, scaling_modifier, override_color)


class HybridGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int):
        """Hybrid Gaussian Model that can combine background and object models. It will support dynamics adjustments of the object offset and size/scaling with methods like set_offset, set_scaling, etc.
        The _xyz_bg, _xyz_obj will store the background and object points respectively, _xyz will store the entire scene as a view of the concatenated background and object points.
        """
        super().__init__(sh_degree)
        self._xyz_bg = torch.empty(0)
        self._features_dc_bg = torch.empty(0)
        self._features_rest_bg = torch.empty(0)
        self._scaling_bg = torch.empty(0)
        self._rotation_bg = torch.empty(0)
        self._opacity_bg = torch.empty(0)

        self._xyz_obj = torch.empty(0)
        self._features_dc_obj = torch.empty(0)
        self._features_rest_obj = torch.empty(0)
        self._scaling_obj = torch.empty(0)
        self._rotation_obj = torch.empty(0)
        self._opacity_obj = torch.empty(0)

        self.object_offset = torch.tensor(
            [0.0, 0.0, 0.0], dtype=torch.float, device="cuda"
        )
        self.object_scaling = torch.tensor(1.0, dtype=torch.float, device="cuda")

        self.bg_len = 0

    def load_ply(self, path_bg, path_obj):
        super().load_ply(path_bg)
        self._xyz_bg = self._xyz
        self._features_dc_bg = self._features_dc
        self._features_rest_bg = self._features_rest
        self._scaling_bg = self._scaling
        self._rotation_bg = self._rotation
        self._opacity_bg = self._opacity

        # # for desktop scene
        # mask = torch.logical_and(self._xyz_bg[:, 2] < 2, self._xyz_bg[:, 2] > -6)
        # air_mask_z = torch.logical_and(self._xyz_bg[:, 2] < 1.5, self._xyz_bg[:, 2] > -5)
        # air_mask_x = self._xyz_bg[:, 0] < 2
        # air_mask_y = torch.logical_and(self._xyz_bg[:, 1] < 7, self._xyz_bg[:, 1] > -5)
        # air_mask = torch.logical_and(air_mask_z, air_mask_x)
        # air_mask = torch.logical_and(air_mask, air_mask_y)
        # mask = torch.logical_and(mask, ~air_mask)
        
        # # for garage scene
        # air_mask_z = torch.logical_and(self._xyz_bg[:, 2] < 2, self._xyz_bg[:, 2] > -2.5)
        # air_mask_y = torch.logical_and(self._xyz_bg[:, 1] < 1, self._xyz_bg[:, 1] > -2.5)
        # air_mask_x = torch.logical_and(self._xyz_bg[:, 0] < 5, self._xyz_bg[:, 0] > -2)
        # air_mask = torch.logical_and(air_mask_z, air_mask_y)
        # air_mask = torch.logical_and(air_mask, air_mask_x)
        # mask = ~air_mask
        # air_mask_y = torch.logical_and(self._xyz_bg[:, 1] < 5, self._xyz_bg[:, 1] > -5)
        # air_mask_x = self._xyz_bg[:, 0] < -2
        # air_mask = torch.logical_and(air_mask_z, air_mask_y)
        # air_mask = torch.logical_and(air_mask, air_mask_x)
        # mask = torch.logical_and(mask, ~air_mask)
        

        # self._xyz_bg = self._xyz_bg[mask]
        # self._features_dc_bg = self._features_dc_bg[mask]
        # self._features_rest_bg = self._features_rest_bg[mask]
        # self._scaling_bg = self._scaling_bg[mask]
        # self._rotation_bg = self._rotation_bg[mask]
        # self._opacity_bg = self._opacity_bg[mask]
        # print(
        #     f"background points before prune: {len(mask)}, background points after prune: {mask.sum().item()}"
        # )
        self.bg_len = len(self._xyz_bg)


        super().load_ply(path_obj)
        self._xyz_obj = self._xyz
        self._features_dc_obj = self._features_dc
        self._features_rest_obj = self._features_rest
        self._scaling_obj = self._scaling
        self._rotation_obj = self._rotation
        self._opacity_obj = self._opacity

        # # for lamborghini
        # mask = torch.linalg.norm(self._xyz_obj[:, :2], dim=1) < 3.5
        # mask = torch.logical_and(mask, self._xyz_obj[:, 0] > -1.7)
        # mask = torch.logical_and(mask, self._xyz_obj[:, 0] < 1.7)
        # mask = torch.logical_and(mask, self._xyz_obj[:, 1] > -4)
        # mask = torch.logical_and(mask, self._xyz_obj[:, 1] < 4)

        # self._xyz_obj = self._xyz_obj[mask]
        # self._features_dc_obj = self._features_dc_obj[mask]
        # self._features_rest_obj = self._features_rest_obj[mask]
        # self._scaling_obj = self._scaling_obj[mask]
        # self._rotation_obj = self._rotation_obj[mask]
        # self._opacity_obj = self._opacity_obj[mask]
        # print(
        #     f"object points before prune: {len(mask)}, object points after prune: {mask.sum().item()}"
        # )


        # concatenate background and object points
        self._xyz = torch.cat((self._xyz_bg, self._xyz_obj), dim=0)
        self._features_dc = torch.cat(
            (self._features_dc_bg, self._features_dc_obj), dim=0
        )
        self._features_rest = torch.cat(
            (self._features_rest_bg, self._features_rest_obj), dim=0
        )
        self._scaling = torch.cat((self._scaling_bg, self._scaling_obj), dim=0)
        self._rotation = torch.cat((self._rotation_bg, self._rotation_obj), dim=0)
        self._opacity = torch.cat((self._opacity_bg, self._opacity_obj), dim=0)

    def set_offset(self, offset):
        self.object_offset[0] = offset[0]
        self.object_offset[1] = offset[1]
        self.object_offset[2] = offset[2]
        self._update()

    def set_scaling(self, scaling):
        self.object_scaling.fill_(scaling)
        self._update()

    def _update(self):
        self._xyz[self.bg_len :] = (
            self._xyz_obj * self.object_scaling + self.object_offset
        )
        self._scaling[self.bg_len :] = self._scaling_obj + torch.log(
            self.object_scaling
        )


if __name__ == "__main__":
    gm = HybridGaussianModel(3)
    gm.load_ply(
        "/home/elijah/Documents/cv_project/weng-wong-project-cv/train.ply",
        "/home/elijah/Documents/cv_project/weng-wong-project-cv/object.ply",
    )
    gm.set_scaling(1.2)
    # w2c = torch.tensor(
    #     [
    #         [-7.0710677e-01, 7.0710677e-01, 0, 0.0],
    #         [4.0824828e-01, 4.0824828e-01, -8.1649655e-01, 0],
    #         [-5.7735026e-01, -5.7735026e-01, -5.7735026e-01, 5.1961527],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]
    # )
    # w2c = np.array([
    #     [0.0, 0.735, 0.0, 0.0],
    #     [1.0, 0, 0, 0],
    #     [0, 0, 0, 0],
    #     [0, 0.2035, 1.9896, 1]], dtype=np.float32
    # ).transpose()
    w2c = np.array(
        [
            [-8.3267e-17, 7.3480e-01, -6.7828e-01, 0.0000e00],
            [1.0000e00, -8.3267e-17, -9.0572e-33, 0.0000e00],
            [-6.1630e-33, -6.7828e-01, -7.3480e-01, 0.0000e00],
            [-6.1630e-33, 2.0348e-01, 1.9896e00, 1.0000e00],
        ],
        dtype=np.float32,
    ).transpose()
    with torch.inference_mode():
        image = gm.render(
            w2c=w2c,
            intrinsics={
                "width": 1024,
                "height": 768,
                "focal_x": 800,
                "focal_y": 800,
            },
        )["render"]

    # import torchvision
    # torchvision.utils.save_image(image, "test_tv.png")

    # image = image.detach().cpu().numpy().transpose(1, 2, 0)
    image = (
        image.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    import cv2

    cv2.imwrite("test.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
