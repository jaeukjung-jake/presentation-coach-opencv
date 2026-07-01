# presentation-coach-opencv

A real-time OpenCV-based presentation coach system that analyzes facial landmarks and provides feedback on gaze, posture, facial expression, and speaking attitude.

This project converts facial landmark points into interpretable geometric features such as distance ratios, face angles, eye states, mouth openness, and facial asymmetry. It also applies normalization, smoothing, and frame-based buffering to improve stability in real-time webcam environments.

---

## Overview

`presentation-coach-opencv` is a real-time computer vision project that analyzes a user's non-verbal presentation attitude using webcam-based facial landmark tracking.

The system detects a face from a live camera frame, tracks facial landmark points, extracts geometric features such as distances, angles, and facial ratios, and provides feedback on presentation habits such as gaze deviation, head tilt, frowning, abnormal eye state, excessive mouth opening, facial asymmetry, and lack of smile.

The main focus of this project is not simply detecting a face, but converting raw landmark coordinates into meaningful feedback that can help users improve their interview or presentation attitude.

---

## Motivation

In interviews and presentations, non-verbal behavior such as eye contact, facial expression, head posture, and mouth movement strongly affects the audience's impression.

However, these habits are difficult to recognize by oneself during practice.  
This project was developed to provide real-time visual feedback by analyzing facial landmarks from webcam input.

The goal was to build a lightweight and interpretable system using C/C++ and OpenCV, while focusing on practical stability in a real-time video environment.

---

## Key Features

### 1. Gaze Deviation Detection

The system estimates whether the user is looking away from the camera by comparing the distance ratio between the nose and both sides of the face.

If the face direction remains biased for multiple frames, the system provides feedback to guide the user to look forward.

### 2. Head Tilt Detection

The roll angle of the face is calculated from the slope between both eyes.

If the user's head is tilted beyond a predefined threshold, the system detects it as poor posture and gives feedback to maintain a straight head position.

### 3. Frown Detection

The system detects a frowning expression by analyzing eyebrow position, eyebrow distance, and the relationship between the eyebrows and eyes.

This helps identify tense or negative facial expressions during a presentation.

### 4. Eye State Analysis

Eye-related landmark ratios are used to detect abnormal eye states such as excessive eye opening or eye closing.

This can indicate nervousness, fatigue, or lack of focus.

### 5. Jaw Openness Detection

The vertical distance between the upper and lower lips is measured to detect excessive mouth opening or unnecessary jaw movement.

This helps the user maintain more natural speaking behavior.

### 6. Facial Asymmetry Detection

The system compares the height and angle difference between the left and right mouth corners.

This helps detect asymmetric mouth shapes, such as an unintended smirk or unbalanced facial expression.

### 7. Smile Feedback

The system analyzes mouth width and smile-related facial ratios to determine whether the user maintains a natural expression.

If the expression remains too rigid, the system encourages the user to make a slight smile.

---

## Technical Highlights

### Geometry-based Feature Analysis

Instead of relying only on a black-box facial expression classifier, this system directly extracts interpretable features from facial landmark coordinates.

The main geometric measurements include:

- Euclidean distance between landmark points
- Angle between both eyes
- Nose-to-side distance ratio
- Eye aspect-related ratio
- Mouth opening distance
- Mouth corner height difference
- Smile-related mouth width ratio

This makes the feedback logic easier to understand, modify, and explain.

---

### Inter-Ocular Distance Normalization

Distance-based measurements can change significantly when the user moves closer to or farther from the webcam.

To reduce this scale variation, distance-based features are normalized using the inter-ocular distance, which is the distance between both eyes.

```cpp
normalized_value = raw_distance / inter_ocular_distance;

## System Pipeline

Webcam Input  
→ Face Detection  
→ Facial Landmark Tracking  
→ Feature Extraction  
→ Feature Normalization  
→ EMA Smoothing  
→ State Buffering  
→ Real-time Feedback Display  

The system first receives live video input from a webcam.  
After detecting the face region, it tracks facial landmark points and extracts geometry-based features such as eye angle, nose-to-face ratio, mouth opening distance, and mouth corner height difference.

The extracted feature values are normalized to reduce scale variation caused by camera distance.  
Then, EMA smoothing is applied to reduce landmark jitter, and frame-based buffering is used to prevent false warnings from short or temporary movements.

Finally, the system displays real-time feedback to help the user improve presentation attitude, gaze, posture, and facial expression.

---

## Project Structure

presentation-coach-opencv/  
├── README.md  
├── project.cpp  
└── include/  
&nbsp;&nbsp;&nbsp;&nbsp;├── ldmarkmodel.h  
&nbsp;&nbsp;&nbsp;&nbsp;└── other header files  

---

## File Description

| File / Folder | Description |
|---|---|
| `project.cpp` | Main source code for webcam input, facial landmark tracking, feature extraction, feedback logic, and real-time display |
| `include/` | Header files required for facial landmark detection and related functions |
| `README.md` | Project documentation |

---

## Tech Stack

- C / C++
- OpenCV
- Webcam-based real-time video processing
- Facial landmark detection
- Geometry-based feature extraction
- Real-time feedback logic

---

## Feedback Categories

| Category | Measurement Method | Feedback Purpose |
|---|---|---|
| Gaze deviation | Nose-to-side distance ratio | Encourage eye contact with the camera |
| Head tilt | Eye slope angle | Maintain straight posture |
| Frown | Eyebrow distance and height | Relax tense facial expressions |
| Eye state | Eye landmark ratio | Detect excessive eye opening or closing |
| Jaw openness | Lip vertical distance | Reduce unnecessary mouth opening |
| Facial asymmetry | Mouth corner height difference | Maintain balanced facial expression |
| No smile | Smile-related mouth ratio | Encourage a natural smile |
