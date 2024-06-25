import pyvista as pv
import numpy as np
import cv2
import os
import json

def generate_camera_intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])

def generate_camera_positions(num_views, radius):
    positions = []
    for i in range(num_views):
        phi = np.pi * (i / num_views)  # Varying from 0 to π
        theta = 2 * np.pi * (i / num_views)  # Varying from 0 to 2π
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        positions.append((x, y, z))
    return positions


def render_images(plotter, camera_positions, intrinsics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    camera_data = []

    for i, position in enumerate(camera_positions):
        try:
            focal_point = [0, 0, 0]
            view_up = [0, 0, 1]

            print(f"Rendering image {i}:")
            print(f"Position: {position}")
            print(f"Focal Point: {focal_point}")
            print(f"View Up: {view_up}")

            plotter.camera_position = [position, focal_point, view_up]
            plotter.camera.zoom(1)
            plotter.render()

            screenshot = plotter.screenshot(transparent_background=False)

            img_name = f'image_{i:05d}.png'
            cv2.imwrite(os.path.join(output_dir, img_name), cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))

            camera_matrix = plotter.camera.GetModelViewTransformMatrix()
            rotation_matrix = [[camera_matrix.GetElement(j, k) for k in range(3)] for j in range(3)]

            camera_data.append({
                "id": i,
                "img_name": img_name,
                "width": screenshot.shape[1],
                "height": screenshot.shape[0],
                "position": list(position),
                "rotation": rotation_matrix,
                "fy": int(intrinsics[1, 1]),
                "fx": int(intrinsics[0, 0])
            })

        except Exception as e:
            print(f"Error rendering image {i}: {e}")
            break

    plotter.close()
    return camera_data

def save_camera_data(camera_data, output_path):
    with open(output_path, 'w') as f:
        json.dump(camera_data, f, indent=4)
