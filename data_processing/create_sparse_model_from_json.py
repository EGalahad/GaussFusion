import json
import sys
import numpy as np
import os
from scipy.spatial.transform import Rotation as R


def quaternion_from_matrix(rotation_matrix):
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()
    return quaternion

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def write_cameras_txt(cameras, output_path):
    with open(output_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam in cameras:
            f.write(f"{cam['id']+1} PINHOLE {cam['width']} {cam['height']} {cam['fx']} {cam['fy']} {cam['width']/2} {cam['height']/2}\n")

def write_images_txt(images, camera_info, output_path):
    # with open(output_path, 'w') as f:
    #     f.write("# Image list with two lines of data per image:\n")
    #     f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    #     f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    #     f.write(f"# Number of images: {len(images)}\n")
        
    #     """The coordinates of the projection/camera center are given by -R^t * T, where R^t is the inverse/transpose of the 3x3 rotation matrix composed from the quaternion and T is the translation vector. """
    #     for img in images:
    #         img['id'] += 1
    #         rotation_matrix = np.array(img['rotation'])
    #         quaternion = quaternion_from_matrix(rotation_matrix)
    #         T = np.array(img['position'])
    #         T = -np.dot(rotation_matrix, T)
    #         f.write(f"{img['id']} {quaternion[3]:.4f} {quaternion[0]:.4f} {quaternion[1]:.4f} {quaternion[2]:.4f} {T[0]:.4f} {T[1]:.4f} {T[2]:.4f} {img['id']} {img['img_name']}\n")
    #         # f.write(f"{img['id']} {quaternion[3]} {quaternion[0]} {quaternion[1]} {quaternion[2]} {T[0]} {T[1]} {T[2]} {img['id']} {img['img_name']}\n")
    #         f.write("\n")
    # return

    camera_extrinsics = {}
    rotate_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    for img in images:
        rotation_matrix = np.array(img['rotation'])
        rotation_matrix = np.dot(rotate_x, rotation_matrix)
        camera_extrinsics[img['img_name']] = (rotation_matrix, img['position'])

    with open(output_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        for img in camera_info:
            rotation_matrix, position = camera_extrinsics[img[1]]
            quaternion = quaternion_from_matrix(rotation_matrix)
            position = -np.dot(rotation_matrix, position)
            f.write(f"{img[0]} {quaternion[3]} {quaternion[0]} {quaternion[1]} {quaternion[2]} {position[0]} {position[1]} {position[2]} {img[0]} {img[1]}\n")
            f.write("\n")
    

def write_points3D_txt(output_path):
    with open(output_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0.0\n")

import sqlite3

def read_camera_info(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute('SELECT image_id, name FROM images ORDER BY image_id')
    camera_info = cursor.fetchall()
    conn.close()
    return camera_info

def main(json_path, database_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "0"), exist_ok=True)
    data = read_json(json_path)
    camera_info = read_camera_info(database_path)

    cameras_txt_path = f"{output_dir}/cameras.txt"
    images_txt_path = f"{output_dir}/images.txt"
    points3D_txt_path = f"{output_dir}/points3D.txt"

    write_cameras_txt(data, cameras_txt_path)
    write_images_txt(data, camera_info, images_txt_path)
    write_points3D_txt(points3D_txt_path)

    print(f"Files have been written to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <json_path> <database path> <output_dir>")
        sys.exit(1)

    json_path = sys.argv[1]
    database_path = sys.argv[2]
    output_dir = sys.argv[3]

    main(json_path, database_path, output_dir)
