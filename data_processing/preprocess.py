from utils import *
import os

def generate_background():
    file_path = 'data/vr_gallery.glb'
    scene_name = "background"
    output_dir = f'{scene_name}/images'
    output_json = f'{scene_name}/output/cameras.json'

    scene = load_model(file_path)
    if not scene:
        return

    mesh = merge_geometries(scene)
    if not mesh:
        return

    pv_mesh = convert_to_pyvista(mesh)
    if not pv_mesh:
        return

    texture = handle_texture(mesh, pv_mesh)

    fx, fy = 800, 800
    cx, cy = 640, 480
    intrinsics = generate_camera_intrinsics(fx, fy, cx, cy)

    num_views = 125
    radius = 10
    camera_positions = generate_camera_positions(num_views, radius)

    camera_data = render_images(pv_mesh, texture, camera_positions, intrinsics, output_dir)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    save_camera_data(camera_data, output_json)


def generate_object():
    file_path = 'data/ficus_bonsai.glb'
    scene_name = "object"
    output_dir = f'{scene_name}/images'
    output_json = f'{scene_name}/output/cameras.json'

    scene = load_model(file_path)
    if not scene:
        return
    
    mesh = merge_geometries(scene)

    pv_mesh = convert_to_pyvista(mesh)
    if not pv_mesh:
        return

    texture = handle_texture(scene, pv_mesh)

    fx, fy = 800, 800
    cx, cy = 640, 480
    intrinsics = generate_camera_intrinsics(fx, fy, cx, cy)

    num_views = 125
    radius = 2
    camera_positions = generate_camera_positions(num_views, radius)

    camera_data = render_images(pv_mesh, texture, camera_positions, intrinsics, output_dir)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    save_camera_data(camera_data, output_json)

if __name__ == "__main__":
    generate_background()
    generate_object()
