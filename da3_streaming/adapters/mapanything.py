import numpy as np
import torch
from typing import List, Optional
from .predictions import PredictionsMA
from mapanything.utils.geometry import depthmap_to_world_frame


class MapAnythingAdapter:
    """
    Adapter to use MapAnything model within DA3-Streaming pipeline.
    Returns PredictionsMA which stores outputs in MapAnything's native format
    for direct reuse in subsequent chunk inference.

    Uses model.forward() directly with native format priors (depth_along_ray,
    ray_directions, cam_quats, cam_trans) for efficient sliding window inference.
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

    def _inject_priors(
        self,
        views: List[dict],
        prior_predictions: PredictionsMA,
        overlap_count: int,
    ) -> List[dict]:
        """
        Inject stored priors in native forward() format into views.

        Injects depth_along_ray, ray_directions, cam_quats, and cam_trans
        from prior predictions directly into the first `overlap_count` views.
        Invalid depths (where mask is False) are set to 0 so the model ignores them.

        Args:
            views: List of view dictionaries from load_images
            prior_predictions: PredictionsMA from previous chunk
            overlap_count: Number of overlap frames at the start

        Returns:
            Modified views with priors injected
        """
        for i in range(overlap_count):
            prior_idx = -overlap_count + i

            # Get native format priors
            depth_along_ray = prior_predictions.depth_along_ray[prior_idx].copy()  # (H, W, 1)
            ray_directions = prior_predictions.ray_directions[prior_idx].copy()  # (H, W, 3)
            cam_quats = prior_predictions.cam_quats[prior_idx].copy()  # (4,)
            cam_trans = prior_predictions.cam_trans[prior_idx].copy()  # (3,)
            mask = prior_predictions.mask[prior_idx]  # (H, W) bool
            # camera_poses = prior_predictions.camera_poses[prior_idx].copy()  # (1, 4, 4)

            # import ipdb; ipdb.set_trace()
            # Zero out invalid depths (model treats depth <= 0 as invalid)
            depth_along_ray[~mask, :] = 0.0

            # Inject in forward() native format
            # These keys are already in the format expected by forward(), so
            # preprocess_input_views_for_inference won't convert them
            views[i]["depth_along_ray"] = torch.from_numpy(depth_along_ray).float().unsqueeze(0).to(self.device)  # (1, H, W, 1)
            views[i]["ray_directions"] = torch.from_numpy(ray_directions).float().unsqueeze(0).to(self.device)  # (1, H, W, 3)
            views[i]["camera_pose_quats"] = torch.from_numpy(cam_quats).float().unsqueeze(0).to(self.device)  # (1, 4)
            views[i]["camera_pose_trans"] = torch.from_numpy(cam_trans).float().unsqueeze(0).to(self.device)  # (1, 3)
            # views[i]["camera_poses"] = torch.from_numpy(camera_poses).float().unsqueeze(0).to(self.device)
            # Remove keys that would conflict (preprocess would try to convert these)
            if "intrinsics" in views[i]:
                del views[i]["intrinsics"]

        return views

    def infer(
        self,
        image_paths: List[str],
        prior_predictions: Optional[PredictionsMA] = None,
        overlap_count: int = 0,
    ) -> PredictionsMA:
        """
        Run inference using model.forward() and return PredictionsMA object.

        Uses native forward() format for priors (depth_along_ray, ray_directions,
        cam_quats, cam_trans) to avoid conversion overhead.

        Args:
            image_paths: List of paths to images
            prior_predictions: Optional PredictionsMA from previous chunk to condition on
            overlap_count: Number of overlap frames at the start that have priors

        Returns:
            PredictionsMA object with native MapAnything format
        """
        from mapanything.utils.image import load_images
        from mapanything.utils.inference import (
            validate_input_views_for_inference,
            preprocess_input_views_for_inference,
            postprocess_model_outputs_for_inference,
        )

        # Load images
        if prior_predictions is not None: #TODO: add and overlap_count > 0
        # if prior_predictions is not None and overlap_count > 0: 
            print(f"Conditioning on {overlap_count} prior frames (native forward() format)")
            # Get stored resolution (H, W) from depth_along_ray and use it for load_images
            # import ipdb; ipdb.set_trace()
            stored_h, stored_w = prior_predictions.depth_along_ray.shape[1:3]
            views = load_images(image_paths, resize_mode="fixed_size", size=(stored_w, stored_h))
        else:
            print(f"Vanilla images-only inference")
            views = load_images(image_paths)

        print(f"Loaded {len(views)} views")

        # Validate input views
        views = validate_input_views_for_inference(views)

        # Inject priors BEFORE preprocessing (priors are already in native format)
        if prior_predictions is not None: #TODO: add and overlap_count > 0
        # if prior_predictions is not None and overlap_count > 0: #TODO: add and overlap_count > 0
            views = self._inject_priors(views, prior_predictions, 10) #TODO: hardcoded for unit testing
            # print(f"Injected {overlap_count} priors in native forward() format")
            print(f"Injected 10 priors in native forward() format")
            # import ipdb; ipdb.set_trace()

        # Preprocess remaining views (converts intrinsics → ray_directions, depth_z → depth_along_ray for non-prior views)
        views = preprocess_input_views_for_inference(views)

        # Move tensors to device
        ignore_keys = {"instance", "idx", "true_shape", "data_norm_type"}
        for view in views:
            for name, val in view.items():
                if name in ignore_keys:
                    continue
                if hasattr(val, "to"):
                    view[name] = val.to(self.device, non_blocking=True)

        # Debug: print shapes before inference
        # if prior_predictions is not None and overlap_count > 0:
        #     print(f"DEBUG: views[0]['img'].shape = {views[0]['img'].shape}")
        #     print(f"DEBUG: views[0]['depth_along_ray'].shape = {views[0]['depth_along_ray'].shape}")
        #     print(f"DEBUG: views[0]['ray_directions'].shape = {views[0]['ray_directions'].shape}")

        # Run forward() with autocast
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model.forward(views, memory_efficient_inference=True)

        # Postprocess outputs (adds img_no_norm, depth_z, intrinsics, camera_poses, mask, conf)
        outputs = postprocess_model_outputs_for_inference(
            raw_outputs=outputs,
            input_views=views,
            apply_mask=True,
            mask_edges=False,
        )

        print("Inference complete!")

        # Extract outputs in both native and derived formats
        depth_along_rays = []
        ray_directions_list = []
        cam_quats_list = []
        cam_trans_list = []
        depths = []
        confs = []
        camera_poses_c2w = []
        extrinsics_w2c = []
        intrinsics = []
        images_out = []
        masks = []
        world_points_list = []

        for pred in outputs:
            # import ipdb; ipdb.set_trace()
            # Native forward() format for future priors
            depth_along_ray = pred["depth_along_ray"][0].cpu().numpy()  # (H, W, 1)
            ray_dirs = pred["ray_directions"][0].cpu().numpy()  # (H, W, 3)
            cam_quat = pred["cam_quats"][0].cpu().numpy()  # (4,)
            cam_tran = pred["cam_trans"][0].cpu().numpy()  # (3,)

            # Derived format for DA3-streaming compatibility
            depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
            depth = depthmap_torch.cpu().numpy()

            camera_pose_c2w_torch = pred["camera_poses"][0]  # (4, 4)
            camera_pose_c2w = camera_pose_c2w_torch.cpu().numpy()

            intrinsic_torch = pred["intrinsics"][0]  # (3, 3)
            intrinsic = intrinsic_torch.cpu().numpy()

            # Compute world points and valid_mask
            world_points, valid_mask = depthmap_to_world_frame(
                depthmap_torch, intrinsic_torch, camera_pose_c2w_torch
            )

            img_no_norm = pred["img_no_norm"][0].cpu().numpy()  # (H, W, 3)
            pred_mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)  # (H, W)

            # Combine prediction mask with valid depth mask
            mask = pred_mask & valid_mask.cpu().numpy()

            # Compute W2C once for DA3-streaming compatibility
            camera_pose_w2c = np.linalg.inv(camera_pose_c2w)[:3, :]  # (3, 4)

            if img_no_norm.max() <= 1.0:
                img = (img_no_norm * 255).astype(np.uint8)
            else:
                img = img_no_norm.astype(np.uint8)

            conf = pred["conf"][0].cpu().numpy()  # (H, W)

            # Append native format
            depth_along_rays.append(depth_along_ray)
            ray_directions_list.append(ray_dirs)
            cam_quats_list.append(cam_quat)
            cam_trans_list.append(cam_tran)

            # Append derived format
            depths.append(depth)
            confs.append(conf)
            camera_poses_c2w.append(camera_pose_c2w)
            extrinsics_w2c.append(camera_pose_w2c)
            intrinsics.append(intrinsic)
            images_out.append(img)
            masks.append(mask)
            world_points_list.append(world_points.cpu().numpy())

        # Stack arrays
        conf_array = np.stack(confs)  # (N, H, W)
        mask_array = np.stack(masks)  # (N, H, W)

        # Normalize confidence across all views to 0-100 range
        conf_min, conf_max = conf_array.min(), conf_array.max()
        if conf_max > conf_min:
            conf_array = (conf_array - conf_min) / (conf_max - conf_min) * 100.0
        else:
            conf_array = np.full_like(conf_array, 50.0)

        # Set confidence to zero for invalid mask regions
        conf_array[~mask_array] = 0.0

        return PredictionsMA(
            # Native forward() format for prior injection
            depth_along_ray=np.stack(depth_along_rays),  # (N, H, W, 1)
            ray_directions=np.stack(ray_directions_list),  # (N, H, W, 3)
            cam_quats=np.stack(cam_quats_list),  # (N, 4)
            cam_trans=np.stack(cam_trans_list),  # (N, 3)
            # Derived format
            depth=np.stack(depths),  # (N, H, W)
            intrinsics=np.stack(intrinsics),  # (N, 3, 3)
            camera_poses_c2w=np.stack(camera_poses_c2w),  # (N, 4, 4)
            conf=conf_array,  # (N, H, W)
            mask=mask_array,  # (N, H, W) bool
            processed_images=np.stack(images_out),  # (N, H, W, 3)
            world_points=np.stack(world_points_list),  # (N, H, W, 3)
            extrinsics=np.stack(extrinsics_w2c),  # (N, 3, 4) - W2C for compatibility
        )
