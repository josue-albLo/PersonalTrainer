# exercise_evaluation.py

import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose


def calculate_angle(point1, point2, point3):
    a = np.array([point1.x, point1.y])
    b = np.array([point2.x, point2.y])
    c = np.array([point3.x, point3.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


def evaluate_squat(landmarks):
    hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

    angle = calculate_angle(hip, knee, ankle)
    ideal_angle = 90
    angle_diff = abs(angle - ideal_angle)
    precision = max(0, 100 - angle_diff)

    return precision


def evaluate_bicep_curl(landmarks):
    shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

    angle = calculate_angle(shoulder, elbow, wrist)
    ideal_angle = 45  # Adjust as per the ideal angle for a bicep curl
    angle_diff = abs(angle - ideal_angle)
    precision = max(0, 100 - angle_diff)

    return precision


def evaluate_jump(landmarks):
    # Define your criteria for a jump here
    # This example uses a simple height check
    hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    base_height = 0.5  # This should be dynamically set based on initial standing height

    height_diff = hip.y - base_height
    precision = max(0, 100 - abs(height_diff * 100))

    return precision


def detect_exercise(landmarks):
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

    # Calculate angles for different exercises
    squat_angle = calculate_angle(left_hip, left_knee, left_ankle)
    bicep_curl_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    # Determine if the exercise is a squat
    if squat_angle < 90:
        return "squat"

    # Determine if the exercise is a bicep curl
    if bicep_curl_angle < 45:
        return "bicep_curl"

    # Determine if the exercise is a jump (example logic)
    # A jump can be detected by significant movement in the y-coordinate of the hip
    if left_hip.y < 0.4:  # Assuming a jump would lift the hip y position significantly
        return "jump"

    return "unknown"
