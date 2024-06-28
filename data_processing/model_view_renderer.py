import pyvista as pv
import numpy as np
import math
import cv2
import os
import json
import tqdm

pv.start_xvfb()

def generate_camera_intrinsics(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def generate_spiral_camera_positions(num_views, radius, rotations):
    positions = []
    height = 0.9 * radius * 2
    height_step = height / num_views
    theta_step = -2 * np.pi * rotations / num_views

    for i in range(num_views):
        theta = i * theta_step
        y = height / 2 - i * height_step
        r = np.sqrt(radius**2 - y**2)
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        positions.append((x, y, z))

    return positions


def render_images(
    plotter, camera_positions, intrinsics, output_dir, focal_point=[0, 0, 0]
):
    plotter.add_light(
        pv.Light(
            position=(5, 5, 5), focal_point=(0, 0, 0), color="white", intensity=1.0
        )
    )
    plotter.add_light(
        pv.Light(
            position=(-5, 5, 5), focal_point=(0, 0, 0), color="white", intensity=0.8
        )
    )
    plotter.add_light(
        pv.Light(
            position=(5, -5, 5), focal_point=(0, 0, 0), color="white", intensity=0.6
        )
    )
    camera_data = []

    view_up = [0, 1, 0]
    for i, position in tqdm.tqdm(enumerate(camera_positions), total=len(camera_positions)):
        try:
            position_offset = [position[0] + focal_point[0], position[1] + focal_point[1], position[2] + focal_point[2]]
            plotter.camera_position = [position_offset, focal_point, view_up]
            plotter.render()

            screenshot = plotter.screenshot(transparent_background=False)

            img_name = f"image_{i:04d}.png"
            cv2.imwrite(
                os.path.join(output_dir, img_name),
                cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR),
            )

            camera_matrix = plotter.camera.GetModelViewTransformMatrix()
            rotation_matrix = [
                [camera_matrix.GetElement(j, k) for k in range(3)] for j in range(3)
            ]

            # reorder the y z coordinate for the position and rotation matrix
            position = [position[0], -position[2], position[1]]
            rotate = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            rotation_matrix = np.dot(rotation_matrix, rotate.T)
            rotation_matrix = rotation_matrix.tolist()

            camera_data.append(
                {
                    "id": i,
                    "img_name": img_name,
                    "width": screenshot.shape[1],
                    "height": screenshot.shape[0],
                    "position": list(position),
                    "rotation": rotation_matrix,
                    "fy": int(intrinsics[1, 1]),
                    "fx": int(intrinsics[0, 0]),
                }
            )

        except Exception as e:
            print(f"Error rendering image {i}: {e}")
            break

    plotter.close()
    return camera_data


def generate_model_views(
    gltf_path: str,
    output_dir: str,
    f: float,
    cx: int,
    cy: float,
    num_views: int,
    radius: float,
    rotations: int,
    focal_point=[0, 0, 0],
):
    images_dir = os.path.join(output_dir, "images")
    output_json = os.path.join(output_dir, "cameras.json")
    os.makedirs(images_dir, exist_ok=True)

    # setup plotter
    w, h = int(2 * cx), int(2 * cy)
    pl = pv.Plotter(off_screen=True, window_size=[w, h])
    pl.import_gltf(gltf_path)

    # intrinsics
    intrinsics = generate_camera_intrinsics(f, cx, cy)
    view_angle = 180 / math.pi * (2.0 * math.atan2(h/2.0, f))
    pl.camera.SetWindowCenter(0, 0)
    pl.camera.SetViewAngle(view_angle)

    # render
    camera_positions = generate_spiral_camera_positions(
        num_views, radius, rotations
    )
    camera_data = render_images(
        pl, camera_positions, intrinsics, images_dir, focal_point
    )
    with open(output_json, "w") as file:
        json.dump(camera_data, file, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate model views for 3D Gaussian Splatting"
    )
    parser.add_argument(
        "-i", "--input_path", type=str, required=True, help="Path to the GLTF file"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Output directory for images, e.g. scene/{scene_name}",
    )
    parser.add_argument("--f", type=float, default=1600, help="Focal length")
    parser.add_argument("--cx", type=float, default=1200, help="Principal point x")
    parser.add_argument("--cy", type=float, default=960, help="Principal point y")
    parser.add_argument(
        "-n",
        "--num_views", type=int, default=100, help="Number of views to generate"
    )
    parser.add_argument(
        "--radius", type=float, default=1.5, help="Radius of the spiral"
    )
    parser.add_argument(
        "--rotations", type=int, default=5, help="Number of rotations in the spiral"
    )
    parser.add_argument(
        "--focal_point",
        type=float,
        nargs=3,
        default=[0, 0, 0],
        help="Focal point for the camera",
    )

    args = parser.parse_args()

    generate_model_views(
        args.input_path,
        args.output_path,
        args.f,
        args.cx,
        args.cy,
        args.num_views,
        args.radius,
        args.rotations,
        args.focal_point,
    )
