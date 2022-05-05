from enum import Enum


WIDTH = 640
HEIGHT = 480
DEPTH_PROJECTION_THRESHOLD = 150
DEPTH_PROJECTION_ANGLE_THRESHOLD = 20
DEPTH_PROJECTION_ANGLE_SKIP = 5
MIN_DEPTH_PROJECTION_ANGLE_FOR_NAVIGATION = 5

BORDER_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


class Direction(Enum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    SIDES = "left or right"
    NONE = "no path ahead"
