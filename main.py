import streamlit as st
import subprocess
import imageio
import cv2
import tempfile
import os
from tracking.tracker import Tracker
from team_assigner.two_team import TeamAssigner


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, path, fps=24):
    writer = imageio.get_writer(path, fps=fps)
    for f in frames:
        writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    writer.close()

st.title("Player Tracking + Team Assignment")

uploaded = st.file_uploader("Upload video", type=["mp4", "avi"])

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded.read())
    st.video(tfile.name)

    if st.button("Run Tracking"):
        st.write("Reading video...")
        video_frames = read_video(tfile.name)
        tracker = Tracker(model_path="models/best.pt")
        tracks = tracker.get_objects(video_frames)

        # Assign team colors using first frame that has players
        for frame_dict in tracks['players']:
            if frame_dict:
                team_assigner = TeamAssigner()
                team_assigner.assign_team_color(video_frames[0], frame_dict)
                break

        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(
                    video_frames[frame_num],
                    track['bbox'],
                    player_id
                )
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = tuple(
                    map(int, team_assigner.team_colors[team])
                )

        output_video, s = tracker.draw_annotations(video_frames, tracks)

        # Lưu video output
        os.makedirs("output_videos", exist_ok=True)
        output_path = "output_videos/output.mp4"
        st.write(output_video[0].shape, output_video[0].dtype)
        save_video(output_video, output_path)

        st.success("Done!")
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("Download Output Video", f, "output.mp4")
