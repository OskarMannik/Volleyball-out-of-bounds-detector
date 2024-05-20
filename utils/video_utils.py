import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # or 'Qt5Agg', 'GTK3Agg', depending on what's available on your system
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def apply_masks_to_frame(frame, player_detections):
    # Apply mask over players
    for track_id, bbox in player_detections.items():
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)  # -1 fills the rectangle
    return frame

def save_video(output_video_frames, output_video_path):
    # Check if the list is empty
    if not output_video_frames:
        print("No frames to save.")
        return  # Exit the function if there are no frames

    # Continue with saving the video if frames are present
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

#------------------------------------------------------------------------

def bbox_intersects(box1, box2):
    """Check if two bounding boxes intersect.

    Args:
        box1, box2: Bounding boxes in the format (x1, y1, x2, y2), where
                    (x1, y1) is the top left and (x2, y2) is the bottom right corner.

    Returns:
        bool: True if boxes intersect, False otherwise.
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return x1 < x4 and x2 > x3 and y1 < y4 and y2 > y3


def collect_intersection_frames(video_frames, ball_detections, player_detections, court_bounds, stubs):
    import pickle
    import pandas as pd

    # Load the pickle file
    with open(stubs, 'rb') as f:
        ball_positions = pickle.load(f)
        
    # Create DataFrame
    df_ball_positions = pd.DataFrame([x.get(1, []) for x in ball_positions], columns=['x1', 'y1', 'x2', 'y2'])
    df_ball_positions.interpolate(inplace=True)
    df_ball_positions.bfill(inplace=True)
    df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
    df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1).mean()
    df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

    # Calculate a threshold for significant movement
    threshold = df_ball_positions['delta_y'].std() * 2  # Threshold is 2 standard deviations from mean

    # Detect significant movements as ball hits
    df_ball_positions['ball_hit'] = (df_ball_positions['delta_y'].abs() >= threshold).astype(int)

    # Debug prints
    print("Delta Y Values:", df_ball_positions['delta_y'].describe())
    print("Detected ball hits:", df_ball_positions['ball_hit'].sum())

    # Filtering frames
    relevant_frames = []
    for frame_num, (frame, ball_dict, players_dict) in enumerate(zip(video_frames, ball_detections, player_detections)):
        ball_hit = df_ball_positions.iloc[frame_num].get('ball_hit', 0)
        if ball_hit == 1:
            ball_bbox = ball_dict.get(1)
            if ball_bbox:
                x1, y1, x2, y2 = ball_bbox
                ball_in_bounds = x1 >= court_bounds[0] and x2 <= court_bounds[1] and y1 >= court_bounds[2] and y2 <= court_bounds[3]
                contact_with_player = any(bbox_intersects(ball_bbox, player_bbox) for player_bbox in players_dict.values())
                if not ball_in_bounds and not contact_with_player:
                    relevant_frames.append(frame)

    print("Total frames collected:", len(relevant_frames))
    return relevant_frames

def save_intersection_frames_as_images(intersection_frames, output_folder="intersection_frames"):
    os.makedirs(output_folder, exist_ok=True)
    for idx, frame in enumerate(intersection_frames):
        if not isinstance(frame, np.ndarray):
            print(f"Error: Frame at index {idx} is not a numpy array.")
            continue
        frame_path = os.path.join(output_folder, f"frame_{idx}.png")
        cv2.imwrite(frame_path, frame)


def save_sample_frames(frames, num_samples=10, output_folder="sample_plots"):
    """Save a sample of video frames as images."""
    os.makedirs(output_folder, exist_ok=True)
    sample_indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
    for idx in sample_indices:
        plt.figure()
        plt.imshow(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {idx}")
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, f"frame_{idx}.png"))
        plt.close()

        
    sample_indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
    for idx in sample_indices:
        plt.figure()
        plt.imshow(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {idx}")
        plt.axis('off')
        plt.savefig(f"{output_folder}/frame_{idx}.png")
        plt.close()
        
def is_within_bounds(x, y, xmin, xmax, ymin, ymax):
    """Check if a point (x, y) is within the specified bounds."""
    return xmin <= x <= xmax and ymin <= y <= ymax

def is_bbox_within_bounds(bbox, xmin, xmax, ymin, ymax):
    """Check if a bounding box is within the specified bounds.
    
    Args:
        bbox: Tuple or list in the format (x1, y1, x2, y2) where
              (x1, y1) is the top left corner and
              (x2, y2) is the bottom right corner.
        xmin, xmax, ymin, ymax: Boundaries of the region to check against.
    
    Returns:
        True if the bounding box is entirely within the bounds, False otherwise.
    """
    x1, y1, x2, y2 = bbox
    return (xmin <= x1 <= xmax and xmin <= x2 <= xmax and
            ymin <= y1 <= ymax and ymin <= y2 <= ymax)