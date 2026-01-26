import numpy as np
import torch
from typing import List, Optional
from .predictions import Predictions
from mapanything.utils.geometry import depthmap_to_world_frame

class MapAnythingAdapter:
    """
    Adapter to use MapAnything model within DA3-Streaming pipeline.
    Converts MapAnything outputs to the unified Predictions format.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    def load(self):
        """Load MapAnything model."""
        from mapanything.models import MapAnything
        
        self.model = MapAnything.from_pretrained("facebook/map-anything")
        self.model = self.model.to(self.device)
        self.model.eval()
        print("MapAnything model loaded.")

    
    def infer(
        self,
        image_paths: List[str],
        prior_predictions: Optional[Predictions] = None,
        overlap_count: int = 0,
    ) -> Predictions:
        """
        Run inference and return unified Predictions object.

        Args:
            image_paths: List of paths to images
            prior_predictions: Optional predictions from previous chunk to condition on
            overlap_count: Number of overlap frames at the start that have priors

        Returns:
            Predictions object with W2C extrinsics
        """
        from mapanything.utils.image import load_images, preprocess_inputs
        from PIL import Image
        import torch.nn.functional as F

        # 1. Check if we need to add priors - if so, load raw images for preprocessing
        if prior_predictions is not None and overlap_count > 0:
            print(f"Conditioning on {overlap_count} prior frames")

            # Convert stored W2C (3,4) back to C2W (4,4) for MapAnything input
            prior_w2c = prior_predictions.extrinsics[-overlap_count:]  # (K, 3, 4)
            prior_c2w = []
            for w2c_34 in prior_w2c:
                w2c_44 = np.eye(4)
                w2c_44[:3, :] = w2c_34
                c2w = np.linalg.inv(w2c_44)
                prior_c2w.append(c2w)
            prior_c2w = np.stack(prior_c2w)  # (K, 4, 4)

            prior_intrinsics = prior_predictions.intrinsics[-overlap_count:]  # (K, 3, 3)
            prior_depth = prior_predictions.depth[-overlap_count:]  # (K, H, W)

            # Build raw views with geometric priors for overlap frames
            raw_views = []
            for i, img_path in enumerate(image_paths):
                img = np.array(Image.open(img_path).convert("RGB"))  # (H, W, 3)
                view = {"img": img}

                if i < overlap_count:
                    # Resize depth to match the raw image resolution
                    # Prior depth is at inference resolution, raw image may be different
                    img_h, img_w = img.shape[:2]
                    depth_h, depth_w = prior_depth[i].shape

                    if (img_h, img_w) != (depth_h, depth_w):
                        # Resize depth to match image
                        depth_tensor = torch.from_numpy(prior_depth[i]).float().unsqueeze(0).unsqueeze(0)
                        resized_depth = F.interpolate(depth_tensor, size=(img_h, img_w), mode='bilinear', align_corners=False)
                        resized_depth = resized_depth.squeeze().numpy()

                        # Scale intrinsics to match new resolution
                        scale_x = img_w / depth_w
                        scale_y = img_h / depth_h
                        scaled_intrinsics = prior_intrinsics[i].copy()
                        scaled_intrinsics[0, 0] *= scale_x  # fx
                        scaled_intrinsics[1, 1] *= scale_y  # fy
                        scaled_intrinsics[0, 2] *= scale_x  # cx
                        scaled_intrinsics[1, 2] *= scale_y  # cy

                        view["intrinsics"] = scaled_intrinsics
                        view["depth_z"] = resized_depth
                    else:
                        view["intrinsics"] = prior_intrinsics[i]
                        view["depth_z"] = prior_depth[i]

                    view["camera_poses"] = prior_c2w[i]  # (4, 4)
                    # view["is_metric_scale"] = np.array([True])  # Must be array, not scalar

                raw_views.append(view)

            # Preprocess all views together (handles resizing of images and geometric inputs)
            views = preprocess_inputs(raw_views)
            print(f"Loaded and preprocessed {len(views)} views with {overlap_count} conditioned")
        else:
            # No priors - use standard load_images
            views = load_images(image_paths)
            print(f"Loaded {len(views)} views")

        # 3. Run inference
        with torch.no_grad():
            outputs = self.model.infer(views,
                                    # apply_mask=True, # Apply masking to dense geometry outputs
                                    # mask_edges=True,
                                )

        print("Inference complete!")

        depths = []
        confs = []
        extrinsics = []
        intrinsics = []
        images_out = []
        masks = []
        world_points_list = []

        for view_idx, pred in enumerate(outputs):
            # Extract data from predictions
            depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
            depth = depthmap_torch.cpu().numpy()
            camera_pose_c2w_torch = pred["camera_poses"][0]  # (4, 4)
            intrinsic_torch = pred["intrinsics"][0]  # (3, 3)

            # Compute world points and valid_mask from depth (same as demo_images_only_inference.py)
            world_points, valid_mask = depthmap_to_world_frame(
                depthmap_torch, intrinsic_torch, camera_pose_c2w_torch
            )

            camera_pose_c2w = camera_pose_c2w_torch.cpu().numpy()
            intrinsic = intrinsic_torch.cpu().numpy()

            img_no_norm = pred["img_no_norm"][0].cpu().numpy() # (H, W, 3)
            pred_mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)  # (H, W)

            # Combine prediction mask with valid depth mask
            mask = pred_mask & valid_mask.cpu().numpy()

            # convert extrinsics from C2W to W2C
            camera_pose_w2c = np.linalg.inv(camera_pose_c2w) # (4, 4)
            camera_pose_w2c = camera_pose_w2c[:3, :] # (3, 4)

            if img_no_norm.max() <= 1.0:
                img = (img_no_norm*255).astype(np.uint8)
            else:
                img = img_no_norm.astype(np.uint8)

            # Get raw confidence scores (will normalize across all views later)
            conf = pred["conf"][0].cpu().numpy()  # (H, W)

            depths.append(depth)
            confs.append(conf)
            extrinsics.append(camera_pose_w2c)
            intrinsics.append(intrinsic)
            images_out.append(img)
            masks.append(mask)
            world_points_list.append(world_points.cpu().numpy())

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
            depth=np.stack(depths),           # (N, H, W)
            conf=conf_array,                  # (N, H, W)
            extrinsics=np.stack(extrinsics),  # (N, 3, 4)
            processed_images=np.stack(images_out),  # (N, H, W, 3)
            intrinsics=np.stack(intrinsics),  # (N, 3, 3)
            mask=mask_array,                  # (N, H, W) bool
            world_points=np.stack(world_points_list),  # (N, H, W, 3)
        )