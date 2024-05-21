import cv2
import os
import numpy as np
import pickle
from Tracker import BallTracker, PlayerTracker
from utils.video_utils import (
    read_video, save_video, bbox_intersects, collect_intersection_frames,
    save_intersection_frames_as_images, save_sample_frames, is_within_bounds, is_bbox_within_bounds
)
from utils.court_utils import select_court_corners, draw_court_outline
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'output_videos/'
app.config['FRAMES_FOLDER'] = 'out_of_bounds_no_contact_frames/'


def process_video(file_path):
    video_frames = read_video(file_path)
    
    if not video_frames:
        return "Error: Video did not load correctly or contains no frames.", None
    
    court_corners = select_court_corners(video_frames[0])
    xmin, ymin = min(point[0] for point in court_corners), min(point[1] for point in court_corners)
    xmax, ymax = max(point[0] for point in court_corners), max(point[1] for point in court_corners)
    court_bounds = (xmin, xmax, ymin, ymax)
    
    court_video_frames = [draw_court_outline(frame.copy(), court_corners) for frame in video_frames]

    ball_tracker = BallTracker('models/last.pt')
    player_tracker = PlayerTracker('yolov8x.pt')
    
    ball_detections = ball_tracker.detect_frames(court_video_frames, read_from_stub=False, stub_path="stubs/ball_detections.pkl")
    interpolated_ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=False, stub_path="stubs/player_detections.pkl")
    
    # Collect and save intersection frames
    out_of_bounds_no_contact_frames = collect_intersection_frames(
        video_frames, ball_detections, player_detections, court_bounds,stubs="stubs/ball_detections.pkl" 
    )
    
    if out_of_bounds_no_contact_frames:
        os.makedirs(app.config['FRAMES_FOLDER'], exist_ok=True)
        frame_filenames = save_intersection_frames_as_images(out_of_bounds_no_contact_frames, app.config['FRAMES_FOLDER'])
    
    output_video_frames = ball_tracker.draw_bboxes(court_video_frames, interpolated_ball_detections)
    output_video_frames = player_tracker.draw_bboxes(output_video_frames, player_detections)
    
    output_video_filename = 'masked_output_video.mp4'
    output_video_path = os.path.join(app.config['PROCESSED_FOLDER'], output_video_filename)
    save_video(output_video_frames, output_video_path)
    print(f"Saved processed video to: {output_video_path}")
    
    return output_video_filename


@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No file part'
    file = request.files['video']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        processed_file_path = process_video(file_path)
        print(f"Processed file path: {processed_file_path}")
        return redirect(url_for('show_results', filename=processed_file_path))

@app.route('/results/<filename>')
def show_results(filename):
    frame_filenames = os.listdir(app.config['FRAMES_FOLDER'])
    frame_filenames = [f for f in frame_filenames if f.endswith('.png') or f.endswith('.jpg')]
    return render_template('results.html', output_video_filename=filename, frame_filenames=frame_filenames)

@app.route('/output_videos/<filename>')
def output_videos(filename):
    # Make sure the path to the directory and filename are correct
    directory = app.config['PROCESSED_FOLDER']
    print(f"Serving video file for download: {filename}")
    return send_from_directory(directory, filename, as_attachment=True)


@app.route('/output_videos/frames/<filename>')
def output_frame(filename):
    return send_from_directory(app.config['FRAMES_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)

