# Smart-Vision-Aid


This Python-based project aims to provide an affordable, efficient, and accessible solution to assist visually impaired individuals by leveraging cutting-edge deep learning models for object detection and depth estimation. The core objective is to replace expensive hardware, such as LiDAR or depth vision cameras, with powerful monocular depth estimation models, making the solution practical and cost-effective.

## Key Features

### 1. Object Detection with YOLO v11
- Utilizes the YOLO (You Only Look Once) v11 model to detect and classify objects in an image with high precision and speed.
- Outputs the bounding box, object class, and confidence level for each detected object.

### 2. Depth Estimation with MiDaS Depth Vision Transformer
- Integrates the MiDaS Depth Vision Transformer to estimate the relative depth of objects using a single RGB image.
- Provides depth maps that allow the system to calculate the relative distance of each detected object from the user.

### 3. Position Estimation
- Combines the object detection results with the depth map to determine the spatial location of objects (e.g., "left", "right", "center") relative to the user.

### 4. Audio Assistance with GTTS
- Uses the Google Text-to-Speech (GTTS) library to generate real-time audio descriptions of the surrounding environment.
- Guides the user with concise and clear audio instructions such as:
  - "A chair is 2 steps ahead on your right."
  - "A person is 3 steps ahead in the front."
  - "A table is nearby on your left."

## Workflow

1. **Image Input**:
   - Captures real-time images using a camera (e.g., a smartphone or Raspberry Pi camera).

2. **Object Detection**:
   - Processes the image through YOLO v11 to identify and localize objects.

3. **Depth Estimation**:
   - Runs the same image through the MiDaS Depth Vision Transformer to generate a depth map.

4. **Integration**:
   - Combines object detection and depth data to estimate the relative position and distance of each object.

5. **Audio Output**:
   - Converts the information into human-readable sentences and generates audio output using GTTS.

6. **User Guidance**:
   - Provides real-time instructions to the user for safe navigation or interaction with their environment.
  
## Requirements
- **cv2**
- **matplotlib**
- **numpy**
- **pytorch**
- **cuda installation for gpu support**(optional)
- **ultralytics**
- **gTTS**

## To test the code, clone this repository and run:
```
python combine.py
```

## Advantages

- **Affordability**: Eliminates the need for costly hardware like LiDAR sensors by relying on monocular RGB images.
- **Accessibility**: Designed for real-world use by visually impaired individuals, enhancing their independence and mobility.
- **Customizability**: Easily adaptable to different environments or specific user needs.

## Potential Applications

- Assisting visually impaired individuals in indoor and outdoor environments.
- Enhancing navigation in complex spaces like public transport terminals or crowded areas.
- Providing contextual information about surroundings to improve situational awareness.

This project demonstrates the potential of combining advanced AI models with affordable hardware to create impactful solutions for accessibility and inclusion.
