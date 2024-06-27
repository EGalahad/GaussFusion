import pyvista as pv
import numpy as np
import cv2
import os
import json

pv.start_xvfb()

def generate_camera_intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def generate_spiral_camera_positions(num_views, radius, height, rotations=2):
    positions = []
    height_step = height / num_views
    theta_step = -2 * np.pi * rotations / num_views

    for i in range(num_views):
        theta = i * theta_step
        x = radius * np.cos(theta)
        y = height / 2 - i * height_step
        z = radius * np.sin(theta)
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
    for i, position in enumerate(camera_positions):
        try:
            plotter.camera_position = [position, focal_point, view_up]
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


def save_camera_data(camera_data, output_path):
    with open(output_path, "w") as f:
        json.dump(camera_data, f, indent=4)


def generate_model_views(
    gltf_path: str,
    output_dir: str,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    num_views: int,
    radius: float,
    height: float,
    rotations: int,
):
    images_dir = os.path.join(output_dir, "images")
    output_json = os.path.join(output_dir, "cameras.json")
    os.makedirs(images_dir, exist_ok=True)

    # setup plotter
    pl = pv.Plotter(off_screen=True)
    pl.import_gltf(gltf_path)

    # generate intrinsics and camera positions
    intrinsics = generate_camera_intrinsics(fx, fy, cx, cy)
    camera_positions = generate_spiral_camera_positions(
        num_views, radius, height, rotations
    )

    # render
    camera_data = render_images(
        pl, camera_positions, intrinsics, images_dir, focal_point=[0, -0.3, 0]
    )
    save_camera_data(camera_data, output_json)


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
    parser.add_argument("--fx", type=float, default=800, help="Focal length x")
    parser.add_argument("--fy", type=float, default=800, help="Focal length y")
    parser.add_argument("--cx", type=float, default=640, help="Principal point x")
    parser.add_argument("--cy", type=float, default=480, help="Principal point y")
    parser.add_argument(
        "-n",
        "--num_views", type=int, default=100, help="Number of views to generate"
    )
    parser.add_argument(
        "--radius", type=float, default=1.5, help="Radius of the spiral"
    )
    parser.add_argument("--height", type=float, default=3, help="Height of the spiral")
    parser.add_argument(
        "--rotations", type=int, default=5, help="Number of rotations in the spiral"
    )

    args = parser.parse_args()

    generate_model_views(
        args.input_path,
        args.output_path,
        args.fx,
        args.fy,
        args.cx,
        args.cy,
        args.num_views,
        args.radius,
        args.height,
        args.rotations,
    )
