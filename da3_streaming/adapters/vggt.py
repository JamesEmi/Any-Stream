import numpy as np
import torch
from typing import List
from .predictions import Predictions
from mapanything.utils.geometry import (
    depthmap_to_world_frame,
    recover_pinhole_intrinsics_from_ray_directions,
)


class VGGTAdapter:
    """
    Adapter to use VGGT model within DA3-Streaming pipeline.
    Converts VGGT outputs to the unified Predictions format.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "vggt",
        resolution_set: int = 518,
        patch_size: int = 14,
    ):
        self.device = device
        self.model_name = model_name
        self.resolution_set = resolution_set
        self.patch_size = patch_size
        self.model = None

    def load(self):
        """Load model via MapAnything's init_model_from_config."""
        from mapanything.models import init_model_from_config

        self.model = init_model_from_config(self.model_name, device=self.device)
        self.model.eval()
        print(f"{self.model_name} model loaded.")

    def infer(self, image_paths: List[str]) -> Predictions:
        """
        Run inference and return unified Predictions object.

        Args:
            image_paths: List of paths to images

        Returns:
            Predictions object with W2C extrinsics
        """
        from mapanything.utils.image import load_images
        from mapanything.models.external.vggt.utils.rotation import quat_to_mat

        # 1. Load images (VGGT: 518/patch 14, VGGT-Omega: 512/patch 16), identity norm
        views = load_images(
            image_paths,
            resolution_set=self.resolution_set,
            norm_type="identity",
            patch_size=self.patch_size,
        )
        print(f"Loaded {len(views)} views")

        # 2. Move view tensors to device (load_images returns CPU tensors and
        # VGGT's forward doesn't auto-move). Mirrors loss_of_one_batch_multi_view.
        ignore_keys = {"data_norm_type", "true_shape", "idx", "instance", "label", "dataset"}
        for view in views:
            for name in list(view.keys()):
                if name in ignore_keys:
                    continue
                val = view[name]
                if isinstance(val, torch.Tensor):
                    view[name] = val.to(self.device, non_blocking=True)

        # 3. Run inference
        with torch.no_grad():
            outputs = self.model(views)

        print("Inference complete!")

        depths = []
        confs = []
        extrinsics = []
        intrinsics = []
        images_out = []
        masks = []
        world_points_list = []

        for view_idx, pred in enumerate(outputs):
            # z-depth is the z-component of pts3d_cam
            depthmap_torch = pred["pts3d_cam"][0][..., 2]  # (H, W)
            depth = depthmap_torch.cpu().numpy()

            # Build camera_pose_c2w (4x4) from cam_trans + cam_quats
            cam_trans = pred["cam_trans"][0]  # (3,)
            cam_quats = pred["cam_quats"][0]  # (4,)
            rot_mat = quat_to_mat(cam_quats.unsqueeze(0))[0]  # (3, 3)
            camera_pose_c2w_torch = torch.eye(
                4, device=rot_mat.device, dtype=rot_mat.dtype
            )
            camera_pose_c2w_torch[:3, :3] = rot_mat
            camera_pose_c2w_torch[:3, 3] = cam_trans

            # Recover pinhole intrinsics from the ray directions VGGT emits
            intrinsic_torch = recover_pinhole_intrinsics_from_ray_directions(
                pred["ray_directions"][0]
            )  # (3, 3)

            # Compute world points and valid_mask from depth
            pts3d_computed, valid_mask = depthmap_to_world_frame(
                depthmap_torch, intrinsic_torch, camera_pose_c2w_torch
            )

            camera_pose_c2w = camera_pose_c2w_torch.cpu().numpy()
            intrinsic = intrinsic_torch.cpu().numpy()

            # Reconstruct denormalized RGB from views (identity norm ⇒ [0, 1])
            img_tensor = views[view_idx]["img"][0]  # (C, H, W)
            img_no_norm = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

            # VGGT has no separate non-ambiguous mask; use depth-validity only
            mask = valid_mask.cpu().numpy()

            # convert extrinsics from C2W to W2C
            camera_pose_w2c = np.linalg.inv(camera_pose_c2w)  # (4, 4)
            camera_pose_w2c = camera_pose_w2c[:3, :]  # (3, 4)

            if img_no_norm.max() <= 1.0:
                img = (img_no_norm * 255).astype(np.uint8)
            else:
                img = img_no_norm.astype(np.uint8)

            # Get confidence scores (squeeze in case of trailing dim)
            conf = pred["conf"][0]
            if conf.ndim == 3:
                conf = conf.squeeze(-1)
            conf = conf.cpu().numpy()  # (H, W)

            depths.append(depth)
            confs.append(conf)
            extrinsics.append(camera_pose_w2c)
            intrinsics.append(intrinsic)
            images_out.append(img)
            masks.append(mask)
            world_points_list.append(pts3d_computed.cpu().numpy())  # (H, W, 3)

        # Stack arrays
        conf_array = np.stack(confs)    # (N, H, W)
        mask_array = np.stack(masks)    # (N, H, W)

        # Normalize confidence across all views to 0-100 range
        conf_min, conf_max = conf_array.min(), conf_array.max()
        if conf_max > conf_min:
            conf_array = (conf_array - conf_min) / (conf_max - conf_min) * 100.0
        else:
            conf_array = np.full_like(conf_array, 50.0)

        # Set confidence to zero for invalid mask regions
        conf_array[~mask_array] = 0.0

        return Predictions(
            depth=np.stack(depths),                  # (N, H, W)
            conf=conf_array,                         # (N, H, W)
            extrinsics=np.stack(extrinsics),         # (N, 3, 4)
            intrinsics=np.stack(intrinsics),         # (N, 3, 3)
            processed_images=np.stack(images_out),   # (N, H, W, 3)
            mask=np.stack(masks),                    # (N, H, W) bool
            world_points=np.stack(world_points_list),  # (N, H, W, 3)
        )
