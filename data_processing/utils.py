import pyvista as pv
import trimesh
import numpy as np
import cv2
import os
import json


def load_model(file_path):
    try:
        scene = trimesh.load(file_path)
        print("Model loaded successfully.")
        return scene
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def merge_geometries(scene):
    if isinstance(scene, trimesh.Scene):
        try:
            mesh = trimesh.util.concatenate([geom for geom in scene.geometry.values()])
            return mesh
        except Exception as e:
            print(f"Error merging geometries: {e}")
            return None
    else:
        return scene
    
def rotation_matrix(axis, angle):
    """
    Create a rotation matrix given an axis and angle.
    :param axis: The axis of rotation (a 3-element array).
    :param angle: The angle of rotation in radians.
    :return: A 3x3 rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotation_matrix_4x4(axis, angle):
    """
    Create a 4x4 transformation matrix given an axis and angle.
    :param axis: The axis of rotation (a 3-element array).
    :param angle: The angle of rotation in radians.
    :return: A 4x4 transformation matrix.
    """
    rot_3x3 = rotation_matrix(axis, angle)
    rot_4x4 = np.eye(4)
    rot_4x4[:3, :3] = rot_3x3
    return rot_4x4

def apply_rotation(mesh, axis, angle):
    """
    Apply rotation to the mesh vertices.
    :param mesh: The input mesh.
    :param axis: The axis of rotation (a 3-element array).
    :param angle: The angle of rotation in radians.
    :return: The rotated mesh.
    """
    rot_matrix = rotation_matrix_4x4(axis, angle)
    mesh.apply_transform(rot_matrix)
    return mesh

def convert_to_pyvista(mesh):
    try:
        vertices = mesh.vertices
        faces = mesh.faces
        print(f"Vertices shape: {vertices.shape}")
        print(f"Faces shape: {faces.shape}")
        print(f"First few faces: {faces[:5]}")

        if faces.shape[1] == 3:
            faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])
        elif faces.shape[1] == 4:
            faces = np.hstack([np.full((faces.shape[0], 1), 4), faces])
        else:
            print("Unexpected faces format.")
            return None

        pv_mesh = pv.PolyData(vertices, faces)
        print("Mesh conversion successful.")
        return pv_mesh
    except Exception as e:
        print(f"Error converting mesh: {e}")
        return None


def handle_texture(mesh, pv_mesh):
    texture = None
    try:
        if mesh.visual.kind == 'texture' and mesh.visual.uv is not None:
            uv = mesh.visual.uv
            pv_mesh.point_data['Texture Coordinates'] = uv
            if hasattr(mesh.visual.material, 'image'):
                texture_image = mesh.visual.material.image
                texture_image = np.array(texture_image)
                texture = pv.Texture(texture_image)
        print("Texture handling successful.")
    except Exception as e:
        print(f"Error handling texture: {e}")
    return texture


def generate_camera_intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])


def generate_camera_positions(num_views, radius):
    positions = []
    for i in range(num_views):
        phi = np.arccos(1 - 2 * (i / num_views))
        theta = np.pi * (1 + 5 ** 0.5) * i
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        positions.append((x, y, z))
    return positions

def render_images(pv_mesh, texture, camera_positions, intrinsics, output_dir):
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pv_mesh, texture=texture, show_scalar_bar=False)
    plotter.add_light(pv.Light(position=(5, 5, 5), focal_point=(0, 0, 0), color='white'))

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


def generate_camera_positions_along_wall(num_views, radius, height, wall_distance):
    positions = []
    focal_points = []
    num_angles = 36
    num_height = int(num_views / num_angles)
    
    for i in range(num_height):
        z = height * (i / num_height)
        for j in range(num_angles):
            theta = j * 2 * np.pi / num_angles
            x = wall_distance * np.cos(theta)
            y = wall_distance * np.sin(theta)
            positions.append((x, y, z))
            focal_points.append((0, 0, z))  # Point towards the center or a specific height

    return positions, focal_points

def render_images_bg(pv_mesh, texture, camera_positions, focal_points, intrinsics, output_dir):
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pv_mesh, texture=texture, show_scalar_bar=False)
    plotter.add_light(pv.Light(position=(5, 5, 5), focal_point=(0, 0, 0), color='white'))

    os.makedirs(output_dir, exist_ok=True)

    camera_data = []

    for i, (position, focal_point) in enumerate(zip(camera_positions, focal_points)):
        try:
            view_up = [0, 0, 1]  # Up direction

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
