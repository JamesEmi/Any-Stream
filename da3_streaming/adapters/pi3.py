import numpy as np
import torch
from typing import List
from .predictions import Predictions


class Pi3Adapter:
    """
    Adapter to use Pi3 model within DA3-Streaming pipeline.
    Converts Pi3 outputs to the unified Predictions format.

    Pi3 directly outputs world points, so we use those instead of
    computing them from depth + intrinsics.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    def load(self):
        """Load Pi3 model."""
        from mapanything.models import init_model_from_config

        self.model = init_model_from_config("pi3", device=self.device)
        self.model.eval()
        print("Pi3 model loaded.")

    def infer(self, image_paths: List[str]) -> Predictions:
        """
        Run inference and return unified Predictions object.

        Args:
            image_paths: List of paths to images

        Returns:
            Predictions object with W2C extrinsics and world points
        """
        from mapanything.utils.image import load_images
        from mapanything.models.external.vggt.utils.rotation import quat_to_mat

        # 1. Load images with Pi3's requirements (resolution 518, identity normalization)
        views = load_images(
            image_paths,
            resolution_set=518,
            norm_type="identity",
            patch_size=14,
        )
        print(f"Loaded {len(views)} views")

        # 2. Run inference
        with torch.no_grad():
            outputs = self.model(views)

        print("Inference complete!")

        depths = []
        confs = []
        extrinsics = []
        images_out = []
        masks = []
        world_points_list = []

        for view_idx, pred in enumerate(outputs):
            # Pi3 outputs have shape (B, H, W, ...) where B=1
            # pts3d: world coordinates, pts3d_cam: camera coordinates
            pts3d = pred["pts3d"][0]  # (H, W, 3)
            pts3d_cam = pred["pts3d_cam"][0]  # (H, W, 3)

            # Z-depth is the z-component of pts3d_cam
            depth = pts3d_cam[..., 2].cpu().numpy()  # (H, W)

            # Get camera pose (cam2world) from quaternion and translation
            cam_trans = pred["cam_trans"][0]  # (3,)
            cam_quats = pred["cam_quats"][0]  # (4,)

            # Convert quaternion to rotation matrix
            rot_mat = quat_to_mat(cam_quats.unsqueeze(0))[0]  # (3, 3)

            # Build C2W matrix
            camera_pose_c2w = torch.eye(4, device=rot_mat.device, dtype=rot_mat.dtype)
            camera_pose_c2w[:3, :3] = rot_mat
            camera_pose_c2w[:3, 3] = cam_trans

            # Convert to W2C
            camera_pose_w2c = torch.linalg.inv(camera_pose_c2w)
            camera_pose_w2c = camera_pose_w2c[:3, :].cpu().numpy()  # (3, 4)

            # Get confidence scores
            conf = pred["conf"][0]  # (H, W) or (H, W, 1)
            if conf.ndim == 3:
                conf = conf.squeeze(-1)
            conf = conf.cpu().numpy()

            # Create mask from valid depth and world points
            world_pts = pts3d.cpu().numpy()  # (H, W, 3)
            mask = (depth > 0) & np.isfinite(depth) & np.all(np.isfinite(world_pts), axis=-1)

            # Get the original image from views (identity normalization = [0, 1])
            img_tensor = views[view_idx]["img"][0]  # (C, H, W)
            img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            depths.append(depth)
            confs.append(conf)
            extrinsics.append(camera_pose_w2c)
            images_out.append(img)
            masks.append(mask)
            world_points_list.append(world_pts)

        # Stack arrays
        conf_array = np.stack(confs)    # (N, H, W)
        mask_array = np.stack(masks)    # (N, H, W)

        # Normalize confidence across all views to 0-100 range
        conf_min, conf_max = conf_array.min(), conf_array.max()
        conf_array = (conf_array - conf_min) / (conf_max - conf_min) * 100.0

        # Set confidence to zero for invalid mask regions
        conf_array[~mask_array] = 0.0

        return Predictions(
            depth=np.stack(depths),              # (N, H, W)
            conf=conf_array,                     # (N, H, W)
            extrinsics=np.stack(extrinsics),     # (N, 3, 4)
            intrinsics=None,                     # Not needed - we have world_points
            processed_images=np.stack(images_out),  # (N, H, W, 3)
            mask=mask_array,                     # (N, H, W) bool
            world_points=np.stack(world_points_list),  # (N, H, W, 3)
        )