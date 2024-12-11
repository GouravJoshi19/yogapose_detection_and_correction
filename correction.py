import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points a, b, and c.
    The angle is calculated as the angle between the vector from b to a and b to c.
    """
    # Vectors from b to a and b to c
    vector_ab = a - b
    vector_bc = c - b

    # Dot product of the vectors
    dot_product = np.dot(vector_ab, vector_bc)

    # Magnitude of the vectors
    norm_ab = np.linalg.norm(vector_ab)
    norm_bc = np.linalg.norm(vector_bc)

    # Calculate the cosine of the angle
    cosine_angle = dot_product / (norm_ab * norm_bc)

    # Clamp value to avoid numerical issues leading to values > 1 or < -1
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate the angle in radians and then convert to degrees
    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)

    return angle_degrees


def analyze_warrior2_pose(landmarks):
    """
    Analyze the Warrior 2 pose based on the landmarks provided.
    Calculate the angles for the elbows, knees, and the straightness of the arms.
    """
    # Extract relevant landmarks (indexing according to Mediapipe's pose landmarks)
    left_shoulder = np.array([landmarks[11], landmarks[12], landmarks[13]])
    left_elbow = np.array([landmarks[14], landmarks[15], landmarks[16]])
    left_wrist = np.array([landmarks[17], landmarks[18], landmarks[19]])

    right_shoulder = np.array([landmarks[5], landmarks[6], landmarks[7]])
    right_elbow = np.array([landmarks[8], landmarks[9], landmarks[10]])
    right_wrist = np.array([landmarks[11], landmarks[12], landmarks[13]])

    left_hip = np.array([landmarks[23], landmarks[24], landmarks[25]])
    left_knee = np.array([landmarks[26], landmarks[27], landmarks[28]])
    left_ankle = np.array([landmarks[29], landmarks[30], landmarks[31]])

    right_hip = np.array([landmarks[11], landmarks[12], landmarks[13]])
    right_knee = np.array([landmarks[14], landmarks[15], landmarks[16]])
    right_ankle = np.array([landmarks[17], landmarks[18], landmarks[19]])

    # Calculate angles for the elbows
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Calculate angles for the knees
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # Feedback based on angles
    feedback = []

    # Check if the elbows are straight (expected angle is around 180 degrees)
    if not (160 <= left_elbow_angle <= 180):
        feedback.append("Left elbow is not straight. Extend your arm more.")
    if not (160 <= right_elbow_angle <= 180):
        feedback.append("Right elbow is not straight. Extend your arm more.")

    # Check if the knees are bent correctly (expected angle is around 120-150 degrees for Warrior 2)
    if not (120 <= left_knee_angle <= 150):
        feedback.append("Left knee angle is not correct. Try bending your knee more.")
    if not (120 <= right_knee_angle <= 150):
        feedback.append("Right knee angle is not correct. Try bending your knee more.")

    # Check for the straightness of the arms (they should be parallel to the ground)
    if not (left_elbow_angle > 160 and right_elbow_angle > 160):
        feedback.append("Both arms should be straight and parallel to the ground.")

    # If no issues with the angles, give positive feedback
    if not feedback:
        feedback.append("Your Warrior 2 pose looks good! Keep it up!")

    return "\n".join(feedback)


def analyze_tree_pose(landmarks):
    """
    Analyze the Tree Pose (Vrksasana) based on the landmarks provided.
    Calculate the angles for the knees, hips, and arms.
    """
    # Extract relevant landmarks (indexing according to Mediapipe's pose landmarks)
    left_hip = np.array([landmarks[23], landmarks[24], landmarks[25]])
    left_knee = np.array([landmarks[26], landmarks[27], landmarks[28]])
    left_ankle = np.array([landmarks[29], landmarks[30], landmarks[31]])

    right_hip = np.array([landmarks[11], landmarks[12], landmarks[13]])
    right_knee = np.array([landmarks[14], landmarks[15], landmarks[16]])
    right_ankle = np.array([landmarks[17], landmarks[18], landmarks[19]])

    left_shoulder = np.array([landmarks[11], landmarks[12], landmarks[13]])
    left_elbow = np.array([landmarks[14], landmarks[15], landmarks[16]])
    left_wrist = np.array([landmarks[17], landmarks[18], landmarks[19]])

    right_shoulder = np.array([landmarks[5], landmarks[6], landmarks[7]])
    right_elbow = np.array([landmarks[8], landmarks[9], landmarks[10]])
    right_wrist = np.array([landmarks[11], landmarks[12], landmarks[13]])

    # Calculate angles for the knees, hips, and arms
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Feedback based on angles
    feedback = []

    # Check for correct knee positioning (one leg straight and the other bent)
    # Stance leg should be straight (knee angle between 170-180 degrees)
    if not (170 <= left_knee_angle <= 180):  # Left stance leg should be straight
        feedback.append("Left stance leg should be straight.")
    if not (170 <= right_knee_angle <= 180):  # Right stance leg should be straight
        feedback.append("Right stance leg should be straight.")

    # Bent leg knee should be around 60-90 degrees (foot resting on the inner thigh or calf of the stance leg)
    if not (60 <= left_knee_angle <= 90):  # Left knee should be bent
        feedback.append("Left knee should be bent at an angle between 60-90 degrees.")
    if not (60 <= right_knee_angle <= 90):  # Right knee should be bent
        feedback.append("Right knee should be bent at an angle between 60-90 degrees.")

    # Check hips: Hips should be level (both should be aligned and parallel to the ground)
    if not (160 <= left_hip_angle <= 180):
        feedback.append("Left hip should be aligned forward and level.")
    if not (160 <= right_hip_angle <= 180):
        feedback.append("Right hip should be aligned forward and level.")

    # Check arm positioning: Both arms should be raised overhead in a Namaste position
    if not (160 <= left_elbow_angle <= 180):  # Elbow should be slightly bent, not completely straight
        feedback.append("Left arm should be raised, elbows slightly bent like in Namaste.")
    if not (160 <= right_elbow_angle <= 180):  # Elbow should be slightly bent, not completely straight
        feedback.append("Right arm should be raised, elbows slightly bent like in Namaste.")

    # Check if the body is upright: The torso should be straight and balanced (the head should be aligned with the body)
    # A body alignment check can be added based on the shoulders or other landmarks, but for simplicity, we'll assume the user is balancing well
    if abs(left_shoulder[1] - right_shoulder[1]) > 0.1:  # Small deviation threshold
        feedback.append("Your body seems tilted. Keep your torso upright and balanced.")

    # Positive feedback if no issues found
    if not feedback:
        feedback.append("Your Tree Pose looks great! Stay balanced and centered.")

    return "\n".join(feedback)

def analyze_goddess_pose(landmarks):
    """
    Analyze the Goddess Pose (Utkata Konasana) based on the landmarks provided.
    Calculate the angles for the knees, hips, and arms.
    """
    # Extract relevant landmarks (indexing according to Mediapipe's pose landmarks)
    left_hip = np.array([landmarks[23], landmarks[24], landmarks[25]])
    left_knee = np.array([landmarks[26], landmarks[27], landmarks[28]])
    left_ankle = np.array([landmarks[29], landmarks[30], landmarks[31]])

    right_hip = np.array([landmarks[11], landmarks[12], landmarks[13]])
    right_knee = np.array([landmarks[14], landmarks[15], landmarks[16]])
    right_ankle = np.array([landmarks[17], landmarks[18], landmarks[19]])

    left_shoulder = np.array([landmarks[11], landmarks[12], landmarks[13]])
    left_elbow = np.array([landmarks[14], landmarks[15], landmarks[16]])
    left_wrist = np.array([landmarks[17], landmarks[18], landmarks[19]])

    right_shoulder = np.array([landmarks[5], landmarks[6], landmarks[7]])
    right_elbow = np.array([landmarks[8], landmarks[9], landmarks[10]])
    right_wrist = np.array([landmarks[11], landmarks[12], landmarks[13]])

    # Calculate angles for the knees, hips, and arms
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Feedback based on angles
    feedback = []

    # Check for knee bend (knees should be at 90-120 degrees)
    if not (90 <= left_knee_angle <= 120):
        feedback.append("Left knee is not bent enough. Bend your knee deeper.")
    if not (90 <= right_knee_angle <= 120):
        feedback.append("Right knee is not bent enough. Bend your knee deeper.")

    # Check hip alignment (hips should be lower than the knees, indicating a squat)
    if not (160 <= left_hip_angle <= 180):
        feedback.append("Left hip should be lower to match the knee bend.")
    if not (160 <= right_hip_angle <= 180):
        feedback.append("Right hip should be lower to match the knee bend.")

    # Check arm position (arms should be at shoulder height, elbows slightly bent)
    if not (160 <= left_elbow_angle <= 180):  # Elbow slightly bent
        feedback.append("Left arm should be extended with a slight bend in the elbow.")
    if not (160 <= right_elbow_angle <= 180):  # Elbow slightly bent
        feedback.append("Right arm should be extended with a slight bend in the elbow.")

    # Check for the torso alignment (the body should remain straight and the chest open)
    if abs(left_shoulder[1] - right_shoulder[1]) > 0.1:  # Small deviation threshold for leaning
        feedback.append("Your torso seems tilted. Keep your body upright and shoulders open.")
    
    # Positive feedback if no issues found
    if not feedback:
        feedback.append("Your Goddess Pose looks great! Keep it strong and balanced.")

    return "\n".join(feedback)

def analyze_plank_pose(landmarks):
    """
    Analyze the Plank Pose based on the landmarks provided.
    Calculate the angles for the body, arms, and hips.
    """
    # Extract relevant landmarks (indexing according to Mediapipe's pose landmarks)
    left_shoulder = np.array([landmarks[11], landmarks[12], landmarks[13]])
    right_shoulder = np.array([landmarks[5], landmarks[6], landmarks[7]])
    left_elbow = np.array([landmarks[14], landmarks[15], landmarks[16]])
    right_elbow = np.array([landmarks[8], landmarks[9], landmarks[10]])
    left_wrist = np.array([landmarks[17], landmarks[18], landmarks[19]])
    right_wrist = np.array([landmarks[11], landmarks[12], landmarks[13]])

    left_hip = np.array([landmarks[23], landmarks[24], landmarks[25]])
    right_hip = np.array([landmarks[11], landmarks[12], landmarks[13]])

    # Calculate angles for the arms and body alignment
    shoulder_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)  # Elbow should be 180 degrees
    elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    body_angle = calculate_angle(left_shoulder, left_hip, right_hip)

    # Feedback based on angles
    feedback = []

    # Check for arm alignment (elbows should be under the shoulders and arms straight)
    if not (170 <= elbow_angle <= 180):
        feedback.append("Left arm is not fully extended. Keep your arm straight.")
    if not (170 <= shoulder_angle <= 180):
        feedback.append("Right arm is not fully extended. Keep your arm straight.")

    # Check for body alignment (body should be in a straight line)
    if not (170 <= body_angle <= 180):  # The body should form a straight line
        feedback.append("Your body is not in a straight line. Keep your body aligned.")

    # Check for hips (hips should not sag or rise)
    if left_hip[1] < right_hip[1] - 0.05:  # Slight deviation
        feedback.append("Your hips are sagging. Engage your core to lift your hips.")

    # Positive feedback if no issues found
    if not feedback:
        feedback.append("Your Plank Pose looks great! Keep your core engaged.")

    return "\n".join(feedback)

def analyze_downdog_pose(landmarks):
    """
    Analyze the Downward Dog Pose based on the landmarks provided.
    Calculate the angles for the body, arms, legs, and hips.
    """
    # Extract relevant landmarks (indexing according to Mediapipe's pose landmarks)
    left_shoulder = np.array([landmarks[11], landmarks[12], landmarks[13]])
    right_shoulder = np.array([landmarks[5], landmarks[6], landmarks[7]])
    left_elbow = np.array([landmarks[14], landmarks[15], landmarks[16]])
    right_elbow = np.array([landmarks[8], landmarks[9], landmarks[10]])
    left_wrist = np.array([landmarks[17], landmarks[18], landmarks[19]])
    right_wrist = np.array([landmarks[11], landmarks[12], landmarks[13]])

    left_hip = np.array([landmarks[23], landmarks[24], landmarks[25]])
    right_hip = np.array([landmarks[11], landmarks[12], landmarks[13]])

    left_knee = np.array([landmarks[26], landmarks[27], landmarks[28]])
    right_knee = np.array([landmarks[14], landmarks[15], landmarks[16]])

    # Calculate angles for the arms, hips, and legs
    shoulder_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)  # Should be around 180 degrees
    hip_angle = calculate_angle(left_shoulder, left_hip, right_hip)
    knee_angle = calculate_angle(left_hip, left_knee, right_knee)

    # Feedback based on angles
    feedback = []

    # Check for arm positioning (should be straight)
    if not (170 <= shoulder_angle <= 180):
        feedback.append("Your arms are not straight. Try to extend your arms fully.")

    # Check for correct hip alignment (hips should be raised)
    if not (170 <= hip_angle <= 180):
        feedback.append("Your hips should be higher. Lift your hips towards the ceiling.")

    # Check for knee straightness (legs should be straight, heels towards the floor)
    if not (170 <= knee_angle <= 180):
        feedback.append("Your legs should be straight, with heels pressing towards the floor.")

    # Positive feedback if no issues found
    if not feedback:
        feedback.append("Your Downward Dog Pose looks great! Keep your body aligned.")

    return "\n".join(feedback)