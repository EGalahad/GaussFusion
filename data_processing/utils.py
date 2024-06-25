import pyvista as pv
import numpy as np
import cv2
import os
import json

def generate_camera_intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])

def generate_spiral_camera_positions(num_views, radius, height, rotations=2):
    positions = []
    height_step = height / num_views
    theta_step = 2 * np.pi * rotations / num_views
    
    for i in range(num_views):
        theta = i * theta_step
        x = radius * np.cos(theta)
        y = height / 2 - i * height_step
        z = radius * np.sin(theta)
        positions.append((x, y, z))

    return positions

def render_images(plotter, camera_positions, intrinsics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plotter.add_light(pv.Light(position=(5, 5, 5), focal_point=(0, 0, 0), color='white', intensity=1.0))
    plotter.add_light(pv.Light(position=(-5, 5, 5), focal_point=(0, 0, 0), color='white', intensity=0.8))
    plotter.add_light(pv.Light(position=(5, -5, 5), focal_point=(0, 0, 0), color='white', intensity=0.6))
    camera_data = []

    for i, position in enumerate(camera_positions):
        try:
            focal_point = [0, 0, 0]
            view_up = [0, 1, 0]
            
            if position[1] < 0:
                focal_point[1] = position[1] / 2  

            print(f"Rendering image {i}:")
            print(f"Position: {position}")
            print(f"Focal Point: {focal_point}")
            print(f"View Up: {view_up}")

            plotter.camera_position = [position, focal_point, view_up]
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
