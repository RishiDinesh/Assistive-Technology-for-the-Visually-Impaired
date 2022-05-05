import argparse
import numpy as np
import cv2
from constants import BORDER_COLOR, DEPTH_PROJECTION_THRESHOLD, TEXT_COLOR, WIDTH, Direction
from kinect import KinectSensor
from models.deepface import FacialAnalyser
from models.yoloact import Yolact
from navigate import get_navigation_instructions, get_ray_from_user
import pyttsx3
import time


def main(args):
    sensor_information = {
        "depth_projection_base": np.zeros((DEPTH_PROJECTION_THRESHOLD, WIDTH, 3)),
        "depth_x": WIDTH // 2,
        "depth_y": DEPTH_PROJECTION_THRESHOLD - 1,
        "navigation_direction": Direction.NONE,
        "previous_navigation_direction": Direction.NONE
    }

    # Initialize the models
    kinect_sensor = KinectSensor()
    print("Started!")
    if args.obstacle_detection:
        sensor_information["obstacle_detection_model"] = Yolact()
    if args.facial_analysis:
        sensor_information["facial_analysis_model"] = FacialAnalyser()
    if args.output_mode == "audio":
        sensor_information["speech_synthesizer"] = pyttsx3.init()
        sensor_information["speech_synthesizer"].setProperty("rate", 150)

    while True:
        sensor_information["c_frame"] = kinect_sensor.get_color_image().astype(np.uint8)
        sensor_information["depth_map"] = kinect_sensor.get_depth_map()

        # Perform model analytics
        if args.obstacle_detection:
            sensor_information["obstacles"] = sensor_information["obstacle_detection_model"].detect(
                sensor_information["c_frame"], sensor_information["depth_map"])

        if args.facial_analysis:
            sensor_information["faces"], sensor_information["target_face"] = sensor_information["facial_analysis_model"].analyse(
                sensor_information["c_frame"])

        if args.navigation:
            get_navigation_instructions(sensor_information)

        # Convey information to the user
        if args.output_mode == "display" or args.debug_mode:
            if args.obstacle_detection:
                for ([left, top, width, height], name, priority, depth) in sensor_information["obstacles"]:
                    cv2.rectangle(sensor_information["c_frame"], (left, top), (left+width,
                                                                               top+height), (0, 0, 255), thickness=2)
                    cv2.putText(sensor_information["c_frame"], f"{name} ({depth})", (
                        left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

            if args.facial_analysis:
                cv2.putText(sensor_information["c_frame"], f"Number of people detected: {len(sensor_information['faces'])}", (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)
                for (x, y, w, h), name, features in sensor_information["faces"]:
                    # Display bounding box for face
                    cv2.rectangle(
                        sensor_information["c_frame"], (x, y - 50), (x + w, y + h + 10), BORDER_COLOR, 2)
                    cv2.rectangle(sensor_information["c_frame"], (
                        x - 2, y - 80), (x + (w * 3) // 4, y - 50), BORDER_COLOR, cv2.FILLED)

                    # Display name of detected face
                    cv2.putText(sensor_information["c_frame"], name.capitalize(), (
                        x+2, y-70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)

                    # Display features extracted from face
                    cv2.putText(sensor_information["c_frame"], f"Emotion: {features['emotion']}", (x+2, y-55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)

            if args.navigation and args.show_ray_casts:
                for angle in sensor_information["depth_projection_rays"]:
                    cv2.line(
                        sensor_information["depth_projection"],
                        (sensor_information["depth_x"],
                         sensor_information["depth_y"]),
                        get_ray_from_user(
                            sensor_information["depth_x"], sensor_information["depth_y"], angle, DEPTH_PROJECTION_THRESHOLD),
                        (0, 255, 0),
                        2
                    )

                cv2.imshow("Depth Projection",
                           sensor_information["depth_projection"])

            # Display the views
            cv2.imshow("User's View", sensor_information["c_frame"])
            if args.show_depth_map:
                cv2.imshow("Depth Map", cv2.normalize(sensor_information["depth_map"], 0, 255))

        if args.output_mode == "audio":
            # Obstacle detection instructions
            if args.obstacle_detection:
                for (_, name, priority, depth) in sensor_information["obstacles"]:
                    sensor_information["speech_synthesizer"].say(
                        f"Detected {name} {int(depth)} centimeters away.")

            # Navigation instructions
            if args.navigation:
                if sensor_information["navigation_direction"] != sensor_information["previous_navigation_direction"]:
                    if sensor_information["navigation_direction"] is Direction.NONE:
                        sensor_information["speech_synthesizer"].say(
                            f"Stop! {sensor_information['navigation_direction'].value}")
                    else:
                        sensor_information["speech_synthesizer"].say(
                            f"Move {sensor_information['navigation_direction'].value}!")

            # TODO: Facial Analysis instructions
            if args.facial_analysis:
                for bbox, name, features in sensor_information["faces"]:
                    if features["emotion"] != "neutral":
                        sensor_information["speech_synthesizer"].say(f"{name} is {features['emotion']}")

            sensor_information["speech_synthesizer"].runAndWait()

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-O", "--obstacle-detection", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "-F", "--facial-analysis", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "-N", "--navigation", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "-D", "--show-depth-map", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "-R", "--show-ray-casts", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "-o", "--output-mode", default="audio", choices=["audio", "display"])
    parser.add_argument(
        "-d", "--debug-mode", default = False, action = argparse.BooleanOptionalAction)
    parser.add_argument(
        "-u", "--update-interval", default=5, type=float)

    args = parser.parse_args()

    main(args)
