import pyvista as pv
import trimesh
import numpy as np
import cv2
from PIL import Image
import os
import json

# Load the GLB file using trimesh
try:
    scene = trimesh.load('data/vr_gallery.glb')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Check if the scene contains multiple geometries
if isinstance(scene, trimesh.Scene):
    # Merge all geometries into a single mesh
    try:
        mesh = trimesh.util.concatenate([geom for geom in scene.geometry.values()])
    except Exception as e:
        print(f"Error merging geometries: {e}")
        exit(1)
else:
    mesh = scene

# Convert the trimesh mesh to pyvista PolyData
try:
    vertices = mesh.vertices
    faces = mesh.faces
    print(f"Vertices shape: {vertices.shape}")
    print(f"Faces shape: {faces.shape}")
    print(f"First few faces: {faces[:5]}")

    # Check if faces array has the expected shape and adjust
    if faces.shape[1] == 3:
        # Assuming triangular faces
        faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    elif faces.shape[1] == 4:
        # Assuming quad faces
        faces = np.hstack([np.full((faces.shape[0], 1), 4), faces])
    else:
        print("Unexpected faces format.")
        exit(1)

    pv_mesh = pv.PolyData(vertices, faces)
    print("Mesh conversion successful.")
except Exception as e:
    print(f"Error converting mesh: {e}")
    exit(1)

# Handle texture application
texture = None
try:
    if mesh.visual.kind == 'texture' and mesh.visual.uv is not None:
        uv = mesh.visual.uv
        pv_mesh.point_data['Texture Coordinates'] = uv
        if hasattr(mesh.visual.material, 'image'):
            texture_image = mesh.visual.material.image
            # Convert PIL image to numpy array
            texture_image = np.array(texture_image)
            texture = pv.Texture(texture_image)
    print("Texture handling successful.")
except Exception as e:
    print(f"Error handling texture: {e}")
    exit(1)

# Define a function to generate camera intrinsics matrix
def generate_camera_intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])

# Example camera intrinsics (these values can be adjusted)
fx, fy = 800, 800
cx, cy = 640, 480
intrinsics = generate_camera_intrinsics(fx, fy, cx, cy)

# Define a function to generate camera positions around the model
def generate_camera_positions(num_views, radius):
    positions = []
    for i in range(num_views):
        phi = np.arccos(1 - 2 * (i / num_views))  # Elevation angle
        theta = np.pi * (1 + 5**0.5) * i  # Azimuth angle (golden angle)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        positions.append((x, y, z))
    return positions

# Generate 100 camera positions around the model
num_views = 125
radius = 10  # Adjust the distance as needed
camera_positions = generate_camera_positions(num_views, radius)

# Set up the PyVista plotter for rendering
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(pv_mesh, texture=texture, show_scalar_bar=False)  # Disable scalar bar

# Add lighting to the scene
plotter.add_light(pv.Light(position=(5, 5, 5), focal_point=(0, 0, 0), color='white'))

# Create a directory to save the rendered images
output_dir = 'background_output_images'
os.makedirs(output_dir, exist_ok=True)

# Initialize a list to store camera data for JSON output
camera_data = []

# Render and save images from different camera positions
for i, position in enumerate(camera_positions):
    try:
        # Define camera focal point and view up vector
        focal_point = [0, 0, 0]
        view_up = [0, 0, 1]  # Z-axis direction as view up

        # Debugging information
        print(f"Rendering image {i}:")
        print(f"Position: {position}")
        print(f"Focal Point: {focal_point}")
        print(f"View Up: {view_up}")

        # Set the camera position and orientation
        plotter.camera_position = [position, focal_point, view_up]

        # Adjust zoom to make the model appear larger
        plotter.camera.zoom(1)  # Adjust the zoom factor as needed

        # Render the scene
        plotter.render()

        # Take a screenshot of the current view without the scalar bar
        screenshot = plotter.screenshot(transparent_background=False)

        # Save the image
        img_name = f'image_{i:05d}.png'
        cv2.imwrite(os.path.join(output_dir, img_name), cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))

        # Get the camera rotation matrix
        camera_matrix = plotter.camera.GetModelViewTransformMatrix()
        rotation_matrix = [[camera_matrix.GetElement(j, k) for k in range(3)] for j in range(3)]

        # Collect camera data for JSON output
        camera_data.append({
            "id": i,
            "img_name": img_name,
            "width": screenshot.shape[1],
            "height": screenshot.shape[0],
            "position": list(position),
            "rotation": rotation_matrix,
            "fy": fy,
            "fx": fx
        })

    except Exception as e:
        print(f"Error rendering image {i}: {e}")
        break

# Save the camera data to a JSON file
with open('background_cameras.json', 'w') as f:
    json.dump(camera_data, f, indent=4)

plotter.close()
