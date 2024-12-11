# Yoga Pose Detection and Correction

This project is a comprehensive application for detecting and classifying yoga poses using computer vision and machine learning. It also provides feedback on pose corrections based on Mediapipe's landmark detection. The project includes real-time pose detection via a webcam and supports multiple yoga poses.

---

## Project Features
- **Real-Time Detection**: Detect and classify yoga poses using a webcam.
- **Pose Correction**: Provides feedback to improve your pose based on angle analysis.
- **Interactive Design**: Offers an intuitive interface with pose instructions and real-time feedback.
- **Expandability**: Built with modular code to add more poses or improve corrections in the future.

---

## Supported Yoga Poses
The model is trained to classify and provide corrections for the following yoga poses:
1. **Downdog Pose**  
2. **Goddess Pose**  
3. **Plank Pose**  
4. **Tree Pose**  
5. **Warrior 2 Pose**  

Each pose includes instructions and cautions, displayed interactively in the application.

---

## Model Architecture
The pose detection system is powered by a **Custom Neural Network** trained on a yoga dataset. Below is the architecture:

- **Input Layer**: Preprocessed pose landmarks (33 keypoints, 3 dimensions each).
- **Hidden Layers**: Dense layers with ReLU activation for feature extraction.
- **Output Layer**: Softmax activation to classify poses into one of five categories.

Key Parameters:
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam Optimizer
- **Metrics**: Accuracy

---

## Dataset
The model was trained on a curated dataset containing annotated yoga poses. Key details:
- **Classes**: 5 yoga poses (Downdog, Goddess, Plank, Tree, Warrior 2)
- **Number of Images**: 1000+ images, evenly distributed across classes.
- **Source**: Public datasets combined with custom annotations.

---

## How to Use
### Prerequisites
- Python 3.7+
- Required Libraries: `streamlit`, `mediapipe`, `tensorflow`, `opencv-python`

### Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/GouravJoshi19/yogapose_detection_and_correction.git
   cd yoga-pose-detection
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

## Improving the Model
To further improve the model accuracy and robustness:
1. **Expand Dataset**:
   - Add more annotated images for diverse poses and participants.
2. **Data Augmentation**:
   - Use techniques like rotation, scaling, and flipping to simulate different perspectives.
3. **Fine-Tuning**:
   - Fine-tune the model on a more extensive dataset or use a pre-trained model as a base.
4. **Advanced Pose Correction**:
   - Incorporate dynamic corrections based on user feedback, potentially integrating reinforcement learning for better recommendations.

---

## Challenges and Learning
### Challenges
1. **Learning Mediapipe and Landmark Detection**:
   - As my first advanced computer vision project, understanding Mediapipe's capabilities and limitations was crucial.
2. **Pose Correction Logic**:
   - Hard-coding corrections was challenging but the most effective approach given time constraints.
3. **Debugging Complexities**:
   - Addressing inconsistencies in landmark detections and model predictions required persistent debugging.

### Learnings
- Mastered Mediapipe for pose estimation.
- Built confidence in handling real-time webcam feeds and integrating machine learning models into applications.
- Learned the importance of user feedback loops in improving project usability.

---

## Overall Experience
This project was both challenging and rewarding. Developing a practical application from scratch allowed me to explore computer vision techniques and push my boundaries. I enjoyed experimenting with pose correction algorithms and creating a user-friendly interface.

---

## Screenshots
![Screenshot 2024-12-11 225730](https://github.com/user-attachments/assets/dcd01c4d-0072-43dc-a525-97c9504d13b7)
![Screenshot 2024-12-11 225742](https://github.com/user-attachments/assets/d87662ea-0a4d-45d3-94d2-1e1b4e7766b6)
![Screenshot 2024-12-11 225753](https://github.com/user-attachments/assets/8f8e0f2b-c136-471d-9c69-56b84927e761)
![Screenshot 2024-12-11 225805](https://github.com/user-attachments/assets/13630581-10d1-4034-a653-0d0adf9cab14)
![Screenshot 2024-12-11 225817](https://github.com/user-attachments/assets/f21e2702-c035-4633-bb49-aa165f024d7e)
![Screenshot 2024-12-11 225829](https://github.com/user-attachments/assets/ee5f88ec-948c-4132-817b-9c197f8a6498)



---

Feel free to contribute, raise issues, or suggest improvements!
