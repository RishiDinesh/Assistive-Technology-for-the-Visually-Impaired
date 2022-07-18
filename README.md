# assistive-technology-for-the-visually-impaired

## Aim

The focus of this project is to develop an obstacle detection and avoidance system using the Kinect sensor as a vision substitution device. 
 - The Kinect sensor is chosen due to its ability to capture depth data along with the RGB channel, which can help in constructing a 3D map of the user’s field of view and aid in navigation. 
 - Additionally, it is also equipped with an infrared camera that can be used to enable night vision, providing unhindered performance in both light and dark environments. 
 - The proposed system will also be integrated with a facial recognition and emotion detection module, allowing it to recognize familiar faces in an environment and capture non-verbal cues such as smiles during a conversation, thereby adding some social benefit. 
 - Object detection results, Navigational directions and facial analysis results will be communicated back to the user via an auditory channel in real-time. 
 
 ## Proposed Method

The frame capture of the sensor runs in a loop that is triggered by an init signal. This signal triggers the image acquisition module which captures the appropriate type of image (RGB/IR) along with the depth map. This data, along with the user provided control signal is sent to the core module which executes the appropriate function and outputs the corresponding result. This result, which is in text format is converted to audio and conveyed back to the user.

![image](https://user-images.githubusercontent.com/63601038/179461669-8f1b3714-0ff4-485f-9db0-f11da6cfbb42.png)

## Sensor Used
The hardware for this project involves the Kinect sensor. It consists of an RGB camera that produces images at 640x480 pixels, as well as a depth sensing system, that consists of an IR laser emitter and an IR camera, and produces images at 640x480 pixels. The depth measurement is done using the infrared emitter and camera whereas the computation is done using a patented structured light technique [25]. 
The field of view of the system is 58 degrees horizontal, 45 degrees vertical, 70 degrees diagonal, and the operational range is between 0.8 meters (2.6 ft) and 3.5 meters (11 ft), both of which is determined by the sensor. The frame rate of this sensor is 30 FPS.

![image](https://user-images.githubusercontent.com/63601038/179462432-0a139722-4435-4373-8ba0-543d6f486709.png)

## Results

### Facial Analysis
![image](https://user-images.githubusercontent.com/63601038/179461748-09593c09-881f-4e57-97c9-8db6bc34fc9b.png)

### Obstacle Detection
![image](https://user-images.githubusercontent.com/63601038/179461793-4a071a37-5c74-4eba-a78b-0d73301ae868.png)

### Navigation
![image](https://user-images.githubusercontent.com/63601038/179462841-3d4c5f07-f0fc-410d-940c-146b102ba637.png)
