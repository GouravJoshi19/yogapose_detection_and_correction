# Yoga Pose Detection and Correction System

## Overview
This project is a Yoga Pose Detection and Correction System designed to classify and provide feedback on yoga poses in real-time using a webcam. The system utilizes a combination of computer vision techniques and deep learning models to achieve its goals.

## Approach

### Research
- The project began with research into computer vision techniques for pose estimation and classification.
- Mediapipe, a robust library for landmark detection, was chosen for pose landmark detection due to its accuracy and ease of use.
- Research was conducted to identify the most suitable datasets for yoga pose classification and to design a model architecture that leverages the detected pose landmarks for classification.

### Dataset Collection and Preprocessing
- The dataset was gathered from publicly available yoga pose datasets.
- To address the limited size of the dataset, **data augmentation** techniques such as flipping, rotation, and scaling were applied to increase dataset diversity.
- Instead of training directly on images, the idea of using **landmarks** (coordinates of key body points) was implemented. This allowed the model to learn the positions of key landmarks rather than relying on raw pixel data.

### Model Architecture
- The model is a simple feedforward neural network:
  - Input Layer: Accepts landmark coordinates as input.
  - Hidden Layers: Two dense layers with 64 neurons each, using ReLU activation.
  - Output Layer: A dense layer with softmax activation to classify into five yoga poses.
- This architecture is lightweight, allowing for efficient real-time inference.

### Implementation Steps
1. **Pose Detection**: Using Mediapipe to extract pose landmarks from video frames.
2. **Model Training**:
   - Train the model using preprocessed landmark data.
   - Evaluate the model using metrics such as accuracy, confusion matrix, and classification report.
3. **Real-Time Interface**:
   - Built using Streamlit to provide an intuitive interface for users.
   - Webcam feed integrated to capture and classify poses in real-time.
   - Feedback system to provide corrective instructions for each pose.

## Results
- **Accuracy**: The model achieved an overall accuracy of 85%.
- **Classification Report**:
  | Pose       | Precision | Recall | F1-Score | Support |
  |------------|-----------|--------|----------|---------|
  | Downdog    | 0.96      | 0.95   | 0.95     | 92      |
  | Goddess    | 0.74      | 0.42   | 0.53     | 74      |
  | Plank      | 0.91      | 0.95   | 0.93     | 101     |
  | Tree       | 0.88      | 0.68   | 0.77     | 63      |
  | Warrior 2  | 0.64      | 0.90   | 0.75     | 104     |

- Confusion Matrix:
  ![Confusion Matrix]("./pictures/confusion_matrix.png")

## Challenges
- Learning new techniques such as Mediapipe for pose landmark detection.
- Finding ways to perform pose corrections dynamically, which required significant thought and experimentation.
- Handling bugs during the integration of pose detection with the Streamlit interface.

## Future Enhancements
- **Improved Feedback Mechanism**: Use machine learning models for more accurate pose corrections instead of hard-coded rules.
- **Additional Poses**: Extend the system to detect and provide feedback for more yoga poses.
- **Advanced Models**: Experiment with more complex models such as LSTMs or Transformers for temporal data analysis to improve accuracy and feedback.
- **Mobile Support**: Deploy the application on mobile devices for portability.

## Conclusion
This project provided a great opportunity to delve into computer vision and machine learning. Despite the challenges, it was a rewarding experience to develop a system that combines technology with wellness practices. Looking ahead, this project can be expanded to make yoga practice more accessible and effective for users worldwide.