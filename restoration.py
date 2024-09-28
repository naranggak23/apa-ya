import os
import cv2
import torch
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray


def face_restoration(img, upscale):
    """Run a simple face restoration process."""
    try:
        has_aligned = False
        only_center_face = False
        detection_model = "retinaface_resnet50"

        upscale = upscale if (upscale is not None and upscale > 0) else 2
        upscale = int(upscale)  # convert type to int
        if upscale > 4:  # avoid memory exceeded due to too large upscale
            upscale = 4 
        if max(img.shape[:2]) > 1500:  # avoid memory exceeded due to too large img resolution
            upscale = 1
            has_aligned = False

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
        )

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=5)
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            # align and warp each face
            face_helper.align_warp_face()

        # paste_back
        if not has_aligned:
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = face_helper.paste_faces_to_input_image(draw_box=False)

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return restored_img
    except Exception as error:
        print('Global exception', error)
        return None
