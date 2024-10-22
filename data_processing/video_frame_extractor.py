import cv2
import os


def extract_frames(video_path, images_dir, num_frames):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval between frames to be extracted
    interval = max(1, total_frames // num_frames)

    frame_count = 0
    extracted_count = 0

    while cap.isOpened() and extracted_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            # Write the frame to the output folder
            frame_filename = os.path.join(
                images_dir, f"frame_{extracted_count:04d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_count} frames to {images_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("-i", "--input_path", type=str, help="Path to the video file")
    parser.add_argument(
        "-o", "--output_path", type=str, help="Output folder for extracted frames"
    )
    parser.add_argument(
        "-n",
        "--num_frames",
        type=int,
        default=300,
        help="Number of frames to extract, e.g. 300",
    )
    args = parser.parse_args()

    images_dir = os.path.join(args.output_path, "images")
    os.makedirs(images_dir, exist_ok=True)

    extract_frames(args.input_path, images_dir, args.num_frames)
