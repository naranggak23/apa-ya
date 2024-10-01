import os
import cv2
import copy
import argparse
import insightface
import numpy as np
from PIL import Image
from typing import List, Union

def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model

def getFaceAnalyser(model_path: str, providers, det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser

def get_many_faces(face_analyser, frame: np.ndarray):
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None

def swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame):
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)

def process(source_img: Union[Image.Image, List], target_img: Image.Image, source_indexes: str, target_indexes: str, model: str):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    face_analyser = getFaceAnalyser(model, providers)
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    face_swapper = getFaceSwapModel(model_path)

    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_faces = get_many_faces(face_analyser, target_img)

    if target_faces is not None:
        temp_frame = copy.deepcopy(target_img)
        num_target_faces = len(target_faces)
        num_source_images = len(source_img)

        if isinstance(source_img, list) and num_source_images == num_target_faces:
            for i in range(num_target_faces):
                source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR))
                temp_frame = swap_face(face_swapper, source_faces, target_faces, i, i, temp_frame)
        elif num_source_images == 1:
            source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR))
            num_source_faces = len(source_faces)

            if source_indexes == "-1":
                source_indexes = ','.join(map(lambda x: str(x), range(num_source_faces)))

            source_indexes = source_indexes.split(',')
            for index in range(len(source_indexes)):
                source_index = int(source_indexes[index])
                temp_frame = swap_face(face_swapper, source_faces, target_faces, source_index, index, temp_frame)

        result = temp_frame
    else:
        print("No target faces found!")

    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image

def parse_args():
    parser = argparse.ArgumentParser(description="Face swap.")
    parser.add_argument("--source_img", type=str, required=True, help="The path of source image, it can be multiple images, dir;dir2;dir3.")
    parser.add_argument("--target_img", type=str, required=True, help="The path of target image or folder.")
    parser.add_argument("--output_folder", type=str, required=False, default="temp", help="Folder to save output images.")
    parser.add_argument("--source_indexes", type=str, required=False, default="-1", help="Comma separated list of the face indexes to use (left to right) in the source image, starting at 0 (-1 uses all faces in the source image")
    parser.add_argument("--target_indexes", type=str, required=False, default="-1", help="Comma separated list of the face indexes to swap (left to right) in the target image, starting at 0 (-1 swaps all faces in the target image")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    source_img_paths = args.source_img.split(';')
    target_img_path = args.target_img
    output_folder = args.output_folder

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    source_img = [Image.open(img_path) for img_path in source_img_paths]

    # Check if the target image is a folder
    if os.path.isdir(target_img_path):
        target_images = [os.path.join(target_img_path, file) for file in os.listdir(target_img_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for target_image in target_images:
            target_img = Image.open(target_image)

            model = "./checkpoints/inswapper_128.onnx"
            output_image_path = os.path.join(output_folder, f'result_{os.path.basename(target_image)}')
            result_image = process(source_img, target_img, args.source_indexes, args.target_indexes, model)

            # Save result
            result_image.save(output_image_path)
            print(f'Result saved successfully: {output_image_path}')
    else:
        target_img = Image.open(target_img_path)
        model = "./checkpoints/inswapper_128.onnx"
        output_image_path = os.path.join(output_folder, args.output_img)
        result_image = process(source_img, target_img, args.source_indexes, args.target_indexes, model)

        # Save result
        result_image.save(output_image_path)
        print(f'Result saved successfully: {output_image_path}')
