import numpy as np
import os
import shutil

from colmap_read_write_model import *


def get_camera_directions(R):
    # The camera direction is the negative of the third column of R
    view_direction = -R[:, 2]
    # The up direction is the second column of R
    up_direction = R[:, 1]
    return view_direction, up_direction


def transform_coordinate_system(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    images = read_images_binary(os.path.join(input_folder, "images.bin"))
    points3D = read_points3D_binary(os.path.join(input_folder, "points3D.bin"))
    base_image = min(
        images.values(), key=lambda img: int(os.path.splitext(img.name)[0][-4:])
    )
    R_base = base_image.qvec2rotmat()
    # eye = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # R_base = np.dot(R_base, eye)

    view, up = get_camera_directions(R_base)
    print(f"Base image: {base_image.name}")
    print(f"Rotation matrix: {R_base}")
    print(f"View direction: {view}")
    print(f"Up direction: {up}")
    
    # required rotation: align camera y with world -z and camera x with world x
    R_align = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    
    # The total transformation is R_align * R_base_inv
    R_transform = np.dot(R_align, R_base.T)
    # R_transform = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

    # Transform all images
    transformed_images = {}
    for image_id, image in images.items():
        R_original = qvec2rotmat(image.qvec)
        
        # New rotation: R_new = R_transform * R_original
        R_new = np.dot(R_original, R_transform.T)
        qvec_new = rotmat2qvec(R_new)
        
        # New translation: T_new = R_transform * T_original
        # tvec_new = np.dot(R_transform, image.tvec)
        tvec_new = image.tvec
        
        transformed_images[image_id] = Image(
            id=image.id,
            qvec=qvec_new,
            tvec=tvec_new,
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=image.point3D_ids
        )

    # Transform all 3D points
    transformed_points3D = {}
    for point3D_id, point3D in points3D.items():
        # Transform 3D point coordinates
        xyz_new = np.dot(R_transform, point3D.xyz)

        transformed_points3D[point3D_id] = Point3D(
            id=point3D.id,
            xyz=xyz_new,
            rgb=point3D.rgb,
            error=point3D.error,
            image_ids=point3D.image_ids,
            point2D_idxs=point3D.point2D_idxs
        )


    # Write the transformed images and points to new binary files
    write_images_binary(transformed_images, os.path.join(output_folder, "images.bin"))
    write_points3D_binary(
        transformed_points3D, os.path.join(output_folder, "points3D.bin")
    )

    shutil.copyfile(
        os.path.join(input_folder, "cameras.bin"),
        os.path.join(output_folder, "cameras.bin"),
    )

    print(f"Transformed images and points saved to {output_folder}")
    return transformed_images, transformed_points3D


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        help="Path to the folder that contains images.bin and points3D.bin",
    )
    parser.add_argument(
        "--output_path",
        help="Path to the folder where the transformed images.bin and points3D.bin will be saved",
    )
    args = parser.parse_args()
    transform_coordinate_system(args.input_path, args.output_path)
