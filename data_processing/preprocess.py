from utils import *
import os
import pyvista as pv

def generate_object():
    file_path = 'data/scene.gltf'
    scene_name = "object"
    output_dir = f'{scene_name}/images'
    output_json = f'{scene_name}/output/cameras.json'

    # Set up the plotter
    pl = pv.Plotter(off_screen=True)

    # Import the bonsai GLTF file
    pl.import_gltf(file_path)

    fx, fy = 800, 800
    cx, cy = 640, 480
    intrinsics = generate_camera_intrinsics(fx, fy, cx, cy)

    num_views = 125
    radius = 1
    height = 3
    rotations = 10  # Increase the number of rotations
    camera_positions = generate_spiral_camera_positions(num_views, radius, height, rotations)

    camera_data = render_images(pl, camera_positions, intrinsics, output_dir)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    save_camera_data(camera_data, output_json)

if __name__ == "__main__":
    generate_object()
