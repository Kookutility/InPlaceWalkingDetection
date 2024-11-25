import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

class InPlaceWalkingDetector:
    def __init__(self):
        self.left_ankle_positions = []
        self.right_ankle_positions = []
        self.max_history = 10  # 기록할 프레임 수를 줄여 더 빠르게 인식
        self.position_threshold = 0.05  # 위치 변화 임계값
        self.height_threshold = 1.0  # 높이 변화 임계값
        self.min_height_threshold = 0.1  # 최소 높이 변화 임계값
        self.horizontal_movement_threshold = 0.6  # 수평 이동 임계값
        self.last_in_place_walk_time = 0  # 마지막으로 제자리 걸음이 감지된 시간
        self.in_place_walking_detected = False  # 제자리 걸음이 감지되었는지 여부

    def detect_in_place_walking(self, pose3d):
        if pose3d is None or len(pose3d) < 6:
            return False

        ankle_left = pose3d[0]
        ankle_right = pose3d[5]

        # 상체만 감지되는 상황에서는 ankle 좌표가 추정된 값일 가능성이 높음
        if self.is_upper_body_only(pose3d):
            self.in_place_walking_detected = False  # 상체만 감지된 경우 감지 상태 리셋
            return False

        if not self.is_valid_ankle_position(ankle_left) or not self.is_valid_ankle_position(ankle_right):
            self.in_place_walking_detected = False  # 유효하지 않은 발목 위치일 경우 감지 상태 리셋
            return False

        self.left_ankle_positions.append(ankle_left)
        self.right_ankle_positions.append(ankle_right)

        if len(self.left_ankle_positions) > self.max_history:
            self.left_ankle_positions.pop(0)
            self.right_ankle_positions.pop(0)

        if len(self.left_ankle_positions) < self.max_history:
            return False

        left_ankle_movement = np.linalg.norm(np.max(self.left_ankle_positions, axis=0) - np.min(self.left_ankle_positions, axis=0))
        right_ankle_movement = np.linalg.norm(np.max(self.right_ankle_positions, axis=0) - np.min(self.right_ankle_positions, axis=0))

        left_ankle_height_movement = np.max(self.left_ankle_positions, axis=0)[1] - np.min(self.left_ankle_positions, axis=0)[1]
        right_ankle_height_movement = np.max(self.right_ankle_positions, axis=0)[1] - np.min(self.right_ankle_positions, axis=0)[1]

        left_ankle_horizontal_movement = np.linalg.norm(
            np.max(self.left_ankle_positions, axis=0)[:2] - np.min(self.left_ankle_positions, axis=0)[:2]
        )
        right_ankle_horizontal_movement = np.linalg.norm(
            np.max(self.right_ankle_positions, axis=0)[:2] - np.min(self.right_ankle_positions, axis=0)[:2]
        )

        height_movement = (left_ankle_height_movement + right_ankle_height_movement) / 2
        horizontal_movement = (left_ankle_horizontal_movement + right_ankle_horizontal_movement) / 2

        is_in_place_walking = horizontal_movement < self.horizontal_movement_threshold and self.min_height_threshold < height_movement < self.height_threshold

        self.in_place_walking_detected = is_in_place_walking
        return is_in_place_walking

    def is_upper_body_only(self, pose3d):
        # 상체만 감지되는 상황인지 확인하는 로직 (예: 발목의 Y좌표가 기준보다 높음)
        upper_body_threshold = -0.5  # 이 값을 조정하여 상체/하체의 기준을 설정
        return all(joint[1] > upper_body_threshold for joint in pose3d)

    def is_valid_ankle_position(self, ankle):
        # 발목 위치가 유효한지 확인하는 로직 추가 (예: 임계값 범위 내에 있는지 확인)
        # 여기서는 단순히 모든 값이 0이 아닌 경우를 유효한 값으로 간주
        return not np.all(ankle == 0)

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

# 비디오 파일 경로 설정
video_path = 'your_video_file.mp4'  # 트레드밀 위에서 걷는 영상 파일 경로

# 비디오 캡처 초기화
cap = cv2.VideoCapture(video_path)

# 알고리즘 초기화
detector = InPlaceWalkingDetector()

# 결과 저장을 위한 리스트 초기화
pose3d_list = []
frame_timestamps = []
walking_detection_results = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 프레임의 타임스탬프 얻기
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

    # 이미지를 RGB로 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mediapipe로 포즈 추출
    result = pose.process(image_rgb)

    if result.pose_world_landmarks:
        # 3D 좌표 추출
        landmarks = result.pose_world_landmarks.landmark
        pose3d = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks])
        pose3d_list.append(pose3d)
    else:
        pose3d = None
        pose3d_list.append(None)

    # 알고리즘에 pose3d 입력
    walking_detected = detector.detect_in_place_walking(pose3d)
    walking_detection_results.append(walking_detected)
    frame_timestamps.append(timestamp)

    # 결과를 프레임에 표시 (옵션)
    cv2.putText(frame, f"Walking: {walking_detected}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if walking_detected else (0, 0, 255), 2)

    # 포즈 랜드마크 그리기 (옵션)
    if result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 영상 출력 (옵션)
    cv2.imshow('In-Place Walking Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()

# 결과를 데이터프레임으로 저장
data = pd.DataFrame({
    'timestamp': frame_timestamps,
    'walking_detected': walking_detection_results
})

# 결과를 CSV 파일로 저장
data.to_csv('walking_detection_results.csv', index=False)

# Pose3D 데이터 저장 (필요한 경우)
import pickle
with open('pose3d_data.pkl', 'wb') as f:
    pickle.dump({'timestamps': frame_timestamps, 'pose3d_list': pose3d_list}, f)
