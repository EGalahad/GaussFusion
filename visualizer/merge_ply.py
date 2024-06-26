import numpy as np
from plyfile import PlyData, PlyElement
import os


def merge_ply_files(folder, output_file):
    file_list = [f"{folder}/{file}" for file in os.listdir(folder) if file.endswith(".ply")]
    combined_ply = PlyData.read(file_list[0])
    vertices = combined_ply["vertex"].data

    for file_name in file_list[1:]:
        ply_data = PlyData.read(file_name)
        vertices = np.concatenate((vertices, ply_data["vertex"].data))

    vertex_element = PlyElement.describe(vertices, "vertex")
    elements = [vertex_element]
    PlyData(elements).write(output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    merge_ply_files(args.input_path, args.output_path)
