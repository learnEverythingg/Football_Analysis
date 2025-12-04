import cv2
import pickle
import os
from ultralytics import YOLO
from track import ByteTrack

class Tracker:
    def __init__(self, model_path="models/best.pt"):
        self.model = YOLO(model_path)
        self.tracker = ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_result = self.model.predict(frames[i:i+batch_size], conf=0.7)
            for res in batch_result:
                frame_dets = []
                names = self.model.names
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy()
                for bbox, score, cls_id in zip(boxes, scores, cls_ids):
                    frame_dets.append({
                        "bbox": bbox,
                        "score": float(score),
                        "cls_id": int(cls_id),
                        "cls_name": names[int(cls_id)]
                    })
                detections.append(frame_dets)
        return detections

    def get_objects(self, frames, stub_path=None):
        tracks = {"players": [], "ball": [], "referees": []}
        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})

            for det in detection:
                if det["cls_name"] == "goalkeeper":
                    det["cls_name"] = "player"

            tracked_objects = self.tracker.step(detection)

            for obj in tracked_objects:
                bbox = obj["bbox"].tolist() if hasattr(obj["bbox"], "tolist") else obj["bbox"]
                track_id = obj["id"]
                cls_name = obj.get("cls_name", "player")

                if cls_name == "player":
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_name == "referee":
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                elif cls_name == "ball":
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        x1, y1, x2, y2 = map(int, bbox)
        x_center = int((x1 + x2) / 2)
        width = max(1, x2 - x1)
        color = tuple(map(int, color)) if color is not None else (0, 0, 255)

        cv2.ellipse(frame,
                    center=(x_center, y2),
                    axes=(width, max(1, int(0.35*width))),
                    angle=0,
                    startAngle=-45,
                    endAngle=235,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4)

        if track_id is not None:
            rect_w, rect_h = 40, 20
            x1_rect = x_center - rect_w//2
            x2_rect = x_center + rect_w//2
            y1_rect = (y2 - rect_h//2) + 15
            y2_rect = (y2 + rect_h//2) + 15
            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            cv2.putText(frame, f"{track_id}", (x1_text, y1_rect+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        total_frames = min(len(video_frames), len(tracks["players"]))

        for frame_num in range(total_frames):
            frame = video_frames[frame_num].copy()
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for _, ball in ball_dict.items():
                frame = self.draw_ellipse(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)
        return output_video_frames, total_frames
