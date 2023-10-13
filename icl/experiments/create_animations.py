import os

import cv2
import numpy as np
import typer
from tqdm import tqdm

app = typer.Typer()


@app.command("create")
def create_animation(base_path, output_path, total_duration: float):
    # List subdirectories
    t_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    ts = sorted([int(t[2:]) for t in t_dirs])
    w_dirs = os.listdir(os.path.join(base_path, t_dirs[0]))
    layer_dirs = os.listdir(os.path.join(base_path, t_dirs[0], w_dirs[0]))
    
    print("Times:" + str(ts))
    print("Task-input combinations:" + str(w_dirs))
    print("Layers:", layer_dirs)

    for w_dir in w_dirs:
        # w_path = os.path.join(base_path, w_dir)
        
        for layer in layer_dirs:                    
            img_files = []
            for t in ts:
                img_file = os.path.join(base_path, f"t={t}", w_dir, layer)
                img_files.append((t, img_file))
            
            print("Found image files:")
            # print("\n".join(img_files))
            print(img_files)
            
            # Calculate frame duration
            total_timesteps = ts[-1] - ts[0]
            frame_durations = []
            for i in range(1, len(t_dirs)):
                frame_durations.append((ts[i] - ts[i-1]) / total_timesteps)

            frame_durations.append(frame_durations[-1])

            print("\n")
            print("Frame durations:")
            print(frame_durations)
            
            # Read first image to get dimensions and initialize video writer
            img = cv2.imread(img_files[0][1])
            height, width, layers = img.shape
            output_video_path = os.path.join(output_path, w_dir, f'{layer.split(".png")[0]}.mp4')
            
            if not os.path.exists(os.path.join(output_path, w_dir)):
                os.makedirs(os.path.join(output_path, w_dir))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

            if len(img_files) != len(frame_durations):
                raise ValueError(f"Number of images and frame durations must be equal. Received {len(img_files)} images and {len(frame_durations)} frame durations.") 
            
            # Add frames to the video
            prev_duration = 0.
            prev_frame = 0

            for i in tqdm(range(len(img_files)), desc="Writing frames"):
                img = cv2.imread(img_files[i][1])

                if i == 0:
                    duration = frame_durations[-1] * total_duration
                else:
                    duration = frame_durations[i-1] * total_duration

                curr_duration = prev_duration + duration
                curr_frame = prev_frame + int(curr_duration * 30) - int(prev_duration * 30) 
                frame_count = max(int(duration * 30), 1)  # assuming 30 fps
                prev_duration = curr_duration
                prev_frame = curr_frame

                for _ in range(frame_count):
                    out.write(img)
            
            # Release video writer
            out.release()
            

if __name__ == "__main__":
    app()