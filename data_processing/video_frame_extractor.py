import cv2
import os

def extract_frames(video_path, output_folder, num_frames):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames to {output_folder}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("--input_path", type=str, help="Path to the video file")
    parser.add_argument("--output_path", type=str, help="Output folder for extracted frames")
    parser.add_argument("num_frames", type=int, help="Number of frames to extract")
    args = parser.parse_args()

    extract_frames(args.input_path, args.output_path, args.num_frames)