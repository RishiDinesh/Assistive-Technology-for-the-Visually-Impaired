from typing import Dict, Tuple

import numpy as np

from constants import DEPTH_PROJECTION_ANGLE_SKIP, DEPTH_PROJECTION_THRESHOLD, HEIGHT, MIN_DEPTH_PROJECTION_ANGLE_FOR_NAVIGATION, WIDTH, Direction


def get_depth_projection(sensor_information: Dict) -> None:
    sensor_information["depth_projection"] = np.zeros_like(
        sensor_information["depth_projection_base"])

    depth_map = sensor_information["depth_map"]
    depth_map[depth_map == 0] = np.inf

    closest_obstructions = np.min(depth_map, axis=0)

    for i in range(closest_obstructions.shape[0]):
        if closest_obstructions[i] < DEPTH_PROJECTION_THRESHOLD:
            sensor_information["depth_projection"][DEPTH_PROJECTION_THRESHOLD -
                                                   1 - int(closest_obstructions[i]), i, :] = [255, 255, 255]


def get_ray_from_user(x: int, y: int, angle: int, magnitude: int) -> Tuple[int, int]:
    return int(x + magnitude * np.cos(angle * 0.01745)), int(y - magnitude * np.sin(angle * 0.01745))


def get_depth_projection_rays(sensor_information: Dict) -> None:
    sensor_information["depth_projection_rays"] = []

    for angle in range(0, 181, DEPTH_PROJECTION_ANGLE_SKIP):
        obstacle_exists = False

        for magnitude in range(1, DEPTH_PROJECTION_THRESHOLD):
            xx, yy = get_ray_from_user(
                sensor_information["depth_x"], sensor_information["depth_y"], angle, magnitude)

            if sensor_information["depth_projection"][yy, xx, 0] == 255:
                obstacle_exists = True
                break

        if not obstacle_exists:
            sensor_information["depth_projection_rays"].append(angle)


def get_depth_projection_navigation(sensor_information: Dict) -> None:
    sensor_information["depth_projection_navigation_fields"] = []
    sensor_information["previous_navigation_direction"] = sensor_information["navigation_direction"]
    start_angle = None
    previous_angle = None

    for angle in sensor_information["depth_projection_rays"]:
        if start_angle is None:
            previous_angle = angle
            start_angle = angle
            continue

        if angle - DEPTH_PROJECTION_ANGLE_SKIP == previous_angle:
            previous_angle = angle
        elif previous_angle - start_angle >= MIN_DEPTH_PROJECTION_ANGLE_FOR_NAVIGATION * DEPTH_PROJECTION_ANGLE_SKIP:
            sensor_information["depth_projection_navigation_fields"].append(
                (start_angle, previous_angle))
            start_angle = angle
            previous_angle = angle
        else:
            start_angle = None
            previous_angle = None

    if previous_angle and previous_angle - start_angle >= MIN_DEPTH_PROJECTION_ANGLE_FOR_NAVIGATION * DEPTH_PROJECTION_ANGLE_SKIP:
        sensor_information["depth_projection_navigation_fields"].append(
            (start_angle, previous_angle))

    free_directions = [((end + start) // 2) // 60 for (start, end)
                       in sensor_information["depth_projection_navigation_fields"]]

    if len(free_directions) == 0:
        sensor_information["navigation_direction"] = Direction.NONE
        return

    right_votes = sum([1 for direction in free_directions if direction == 0])
    front_votes = sum([1 for direction in free_directions if direction == 1])
    left_votes = sum([1 for direction in free_directions if direction == 2])

    max_votes = max(right_votes, left_votes, front_votes)

    # Choose direction
    if front_votes == max_votes:
        sensor_information["navigation_direction"] = Direction.FRONT
    else:
        if right_votes == left_votes == max_votes:
            sensor_information["navigation_direction"] = Direction.SIDES
        elif right_votes == max_votes:
            sensor_information["navigation_direction"] = Direction.RIGHT
        else:
            sensor_information["navigation_direction"] = Direction.LEFT


def get_navigation_instructions(sensor_information: Dict) -> None:
    get_depth_projection(sensor_information)
    get_depth_projection_rays(sensor_information)
    get_depth_projection_navigation(sensor_information)
