import numpy as np
import torch
from typing import List
from .predictions import Predictions

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

    
    def infer(self, image_paths: List[str]) -> Predictions:
        """
        Run inference and return unified Predictions object.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            Predictions object with W2C extrinsics
        """
        from mapanything.utils.image import load_images
        
        # 1. Load images
        views = load_images(image_paths)
        print(f"Loaded {len(views)} views")

        # 2. Run inference
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

        for view_idx, pred in enumerate(outputs):
            # Extract data from predictions
            depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
            depth = depthmap_torch.cpu().numpy()
            conf = pred["conf"][0].cpu().numpy()  # Per-pixel confidence scores (H, W)
            camera_pose_c2w = pred["camera_poses"][0].cpu().numpy()  # (4, 4)
            intrinsic_torch = pred["intrinsics"][0]  # (3, 3)
            intrinsic = intrinsic_torch.cpu().numpy()
            
            img_no_norm = pred["img_no_norm"][0].cpu().numpy() # Denormalized input images for visualization (H, W, 3)
            mask = pred["mask"][0].squeeze(-1).cpu().numpy()  # (H, W)

            # convert extrinsics from C2W to W2C
            camera_pose_w2c = np.linalg.inv(camera_pose_c2w) # (4, 4)
            camera_pose_w2c = camera_pose_w2c[:3, :] # (3, 4)

            if img_no_norm.max() <= 1.0:
                img = (img_no_norm*255).astype(np.uint8)
            else:
                img = img_no_norm.astype(np.uint8)

            depths.append(depth)
            confs.append(conf)
            extrinsics.append(camera_pose_w2c)
            intrinsics.append(intrinsic)
            images_out.append(img)
            masks.append(mask)

        return Predictions(
        depth=np.stack(depths),           # (N, H, W)
        conf=np.stack(confs),             # (N, H, W)
        extrinsics=np.stack(extrinsics),  # (N, 3, 4)
        intrinsics=np.stack(intrinsics),  # (N, 3, 3)
        processed_images=np.stack(images_out),  # (N, H, W, 3)
        mask=np.stack(masks),             # (N, H, W)
    )