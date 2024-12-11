import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model  # or appropriate model loading function
import pandas as pd
from correction import calculate_angle,analyze_warrior2_pose,analyze_tree_pose,analyze_goddess_pose,analyze_downdog_pose,analyze_plank_pose

st.set_page_config(page_title="Yoga Pose Detection", layout="wide")


# Initialize Mediapipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load your trained model
MODEL_PATH = "./yoga.keras"  # Replace with the actual path
model = load_model(MODEL_PATH)

POSE_CLASSES = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']
accuracy_path = "./Accuracy_plot (1).png"

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a page", ["Real-Time Detection", "Project Info"])

    if app_mode == "Real-Time Detection":
        run_pose_detection()
    elif app_mode == "Project Info":
        display_project_info()


def run_pose_detection():
    st.title("Real-Time Yoga Pose Detection")
    st.write("This section allows you to perform real-time pose detection using your webcam.")

    # Add details about the model
    st.markdown("""
    ## About the Pose Detection System
    This model is trained to detect and classify the following yoga poses:
""")

    with st.expander("1. Downdog Pose"):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("""
                - **How to Perform**: Start on your hands and knees, lift your hips upward, and straighten your arms and legs to form an inverted 'V' shape.
                - **Cautions**:
                    - Avoid bending your knees excessively.
                    - Do not strain your neck; keep your head relaxed.
                    - Ensure your wrists are supported if you have wrist pain.
            """)
        with col2:
            st.image("./pictures/down_dog.jpg", caption="Downdog Pose", width=150)
    with st.expander("2. Goddess Pose"):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("""
                - **How to Perform**: Stand with feet wide apart, toes turned outward, bend your knees, and bring your thighs parallel to the ground. Extend your arms outward or join palms in front of your chest.
                - **Cautions**:
                    - Avoid overextending the knees.
                    - Keep your back straight and core engaged.
            """)
        with col2:
            st.image("./pictures/goddess_pose.jpeg", caption="Goddess Pose", width=150)

    with st.expander("3. Plank Pose"):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("""
                - **How to Perform**: Start in a push-up position, with your hands under shoulders, legs straight, and body in a straight line from head to heels.
                - **Cautions**:
                    - Avoid sagging hips or lifting them too high.
                    - Keep your core engaged and do not strain your wrists.
            """)
        with col2:
            st.image("./pictures/plank_pose.jpg", caption="Plank Pose", width=150)

    with st.expander("4. Tree Pose"):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("""
                - **How to Perform**: Stand on one leg, bend the other knee, and place the sole of your foot on the inner thigh or calf of the standing leg. Raise your arms above your head and join your palms.
                - **Cautions**:
                    - Avoid placing the foot on the knee joint.
                    - Maintain balance and avoid leaning to one side.
            """)
        with col2:
            st.image("./pictures/tree_pose.jpg", caption="Tree Pose", width=150)

    with st.expander("5. Warrior 2 Pose"):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("""
                - **How to Perform**: Start in a lunge position, with one foot forward and the other extended back. Raise your arms parallel to the ground, aligning with your shoulders.
                - **Cautions**:
                    - Ensure the front knee is directly above the ankle.
                    - Avoid leaning forward or backward; keep your torso upright.
            """)
        with col2:
            st.image("./pictures/warrior2_pose.jpg", caption="Warrior 2 Pose", width=150)

    st.markdown("""
    - It uses a Deep learning model trained on a yoga pose dataset with an accuracy of **85%**.
    - The system provides real-time feedback to improve your posture.
    - To get started, enable your webcam using the options on the sidebar and perform one of the poses above!
""")

    # Display the webcam feed
    st.sidebar.header("Webcam Options")
    run_detection = st.sidebar.checkbox("Start Pose Detection", value=False)

    # Placeholder for webcam output
    frame_placeholder = st.empty()

    if run_detection:
        # Start the webcam feed
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Unable to access webcam. Please check your device.")
            return

        while run_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Unable to read frame from webcam.")
                break

            # Convert BGR to RGB (required for Mediapipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Pose detection
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # Draw landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                # Extract landmarks for prediction
                landmarks = extract_landmarks_from_results(results)

                if landmarks is not None:
                    # Predict pose
                    predicted_pose = model.predict(landmarks.reshape(1, -1))
                    predicted_class = np.argmax(predicted_pose, axis=1)[0]

                    # Map the predicted class index to the pose label
                    predicted_label = POSE_CLASSES[predicted_class]

                    feedback_function_name = f"analyze_{predicted_label}_pose"
                    if feedback_function_name in globals():
                        feedback = globals()[feedback_function_name](landmarks)
                    else:
                        feedback = "Pose not recognized."

                # Display the feedback on the frame
                    feedback_lines = feedback.split("\n")
                    y_offset = 50  # Starting position for feedback text

                    for line in feedback_lines:
                        # Display each line of feedback
                        cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                        y_offset += 30  # Move down for next line

                    # Display the predicted pose on the frame
                    cv2.putText(frame, f"Predicted Pose: {predicted_label}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                else:
                    cv2.putText(frame, "Pose could not be detected.", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Display the frame with pose feedback in the app
            frame_resized = cv2.resize(frame, (4000, 1800))
            frame_placeholder.image(frame_resized, channels="BGR", use_container_width=True)

        cap.release()
    else:
        st.write("Enable the checkbox to start pose detection.")

def extract_landmarks_from_results(results):
    """
    Extract pose landmarks from Mediapipe results, trimmed to match model input.
    """
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            # Use only x, y, and z (excluding visibility)
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks[:99])  # Ensure exactly 99 features
    return None


def display_project_info():
    st.title("Project Information")
    st.markdown(
        """
        ### About the Dataset
        - **Dataset Name**: Yoga Pose Dataset
        - **Source**: [Kaggle Yoga Poses Dataset](https://www.kaggle.com/niharika41298/yoga-poses-dataset)
        - **Classes**: Downdog, Goddess, Plank, Tree, Warrior2
        - **Images per Class**: 200-300

        ### Model Information
        - **Model Architecture**: A fully connected neural network with 3 dense layers.
        - **Accuracy Achieved**: 0.80 on test set
        - **Loss Achieved**:0.52 on the test set.

        ### Steps in Model Building
        1. Data collection and preprocessing (landmark extraction using Mediapipe).
        2. Model training with Keras and TensorFlow.
        3. Evaluation and tuning of hyperparameters.
        """
    )

    # Add training accuracy and loss plots
    st.write("### Training and Validation Metrics")

# Arrange graphs side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image("./pictures/Loss_plot.png",caption="Loss Vs Epoch")
    with col2:
        st.image("./pictures/Accuracy_plot.png",caption='Accuracy Vs Epoch')

    col3,col4=st.columns(2)

    with col3:
        st.image("./pictures/confusion_matrix.png",caption="Confusion Matrix")
    with col4:
        # Create a DataFrame for the classification report
        data = {
            "Class": ["downdog", "goddess", "plank", "tree", "warrior2", "accuracy", "macro avg", "weighted avg"],
            "Precision": [0.98, 0.77, 0.87, 0.77, 0.84, None, 0.84, 0.85],
            "Recall": [0.90, 0.72, 0.97, 0.76, 0.85, None, 0.84, 0.85],
            "F1-Score": [0.94, 0.74, 0.92, 0.77, 0.84, 0.85, 0.84, 0.85],
            "Support": [92, 74, 101, 63, 104, 434, 434, 434],
        }

        # Convert data into a DataFrame
        df = pd.DataFrame(data)

        # Display the table
        st.subheader("Classification Report")
        st.table(df)
        

if __name__ == "__main__":
    main()


