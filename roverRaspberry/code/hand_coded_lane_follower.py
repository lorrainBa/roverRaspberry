import cv2
import numpy as np
import logging
import math
import datetime
import sys
from collections import deque


timeToReact = 20
steerAngleSpeedMemory = deque()
for _ in range (timeToReact):
    steerAngleSpeedMemory.append((90,0))

_SHOW_IMAGE = False
showSlope = False

class HandCodedLaneFollower(object):


    def __init__(self, car=None):
        logging.info('Creating a HandCodedLaneFollower...')
        self.car = car
        self.curr_steering_angle = 90

        
        #Infor if the current line is new or not
        self.leftLineAge = 8
        self.rightLineAge = 8
        
        
    def follow_lane(self, frame, speed):
        # Main entry point of the lane follower
        show_image("orig", frame)

        lane_lines, frame = detect_lane(frame)
        final_frame, stop, newSpeed = self.steer(frame, lane_lines,speed)

        return final_frame,stop,newSpeed

    def steer(self, frame, lane_lines,speed):
        newSpeed = speed
        
        print("Mis à jour angle")
        logging.debug('steering...')
        if len(lane_lines) == 0:
            logging.error('No lane lines detected, nothing to do.')
            return frame,False,0
        #Find the new steering angle
        new_steering_angle = compute_steering_angle(frame, lane_lines)
        self.curr_steering_angle = stabilize_steering_angle(self.curr_steering_angle, new_steering_angle, len(lane_lines),speed)
        
        #Formula to get new Speed, to go forward the angle is 90, if the angle is far from 90 then it slow
        speedChange = speed - int(((90-self.curr_steering_angle))**2/(speed)*1.5)
        
        #Put the new speed and steering angle in the memory
        steerAngleSpeedMemory.append( (self.curr_steering_angle , speedChange) )
        
        if self.car is not None:
            memoryAngle , memorySpeed = steerAngleSpeedMemory.popleft()
            self.car.front_wheels.turn(memoryAngle)
            #Adapt the speed 
            
            
            #Always have 15 as a minimum speed
            if memorySpeed > 20:
                
                if memorySpeed > speed +30:
                    memorySpeed = speed +30
                elif memorySpeed  < speed -30:
                    memorySpeed = speed -30
                else:
                    newSpeed = memorySpeed
            else:
                newSpeed = 20
            print("------------------",newSpeed)
        curr_heading_image = display_heading_line(frame, self.curr_steering_angle)
        show_image("heading", curr_heading_image)
        return curr_heading_image, False, newSpeed




def compute_steering_angle(frame, lane_lines):
    """ Find the steering angle based on lane line coordinate
        We assume that camera is calibrated to point to dead center
    """
    if len(lane_lines) == 0:
        logging.info('No lane lines detected, do nothing')
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        x_offset = x_offset/4
        print("X OFFSET", x_offset)
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        #If left line goes after the right line then one of the line is a false detection and we have to get rid of the new one
        if left_x2 > right_x2:
            if leftLineAge > rightLineAge:
                logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
                x1, _, x2, _ = lane_lines[0][0]
                x_offset = x2 - x1
                x_offset = x_offset/5
                print("X OFFSET", x_offset)
            else:
                logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
                x1, _, x2, _ = lane_lines[1][0]
                x_offset = x2 - x1
                x_offset = x_offset/5
                print("X OFFSET", x_offset)
        else:
            camera_mid_offset_percent = 0.02 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
            mid = int(width / 2 * (1 + camera_mid_offset_percent))
            x_offset = (left_x2 + right_x2) / 2 - mid

    # find the steering angle, which is angle between navigation direction to end of center line
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

    logging.debug('new steering angle: %s' % steering_angle)
    return steering_angle


def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=4, max_angle_deviation_one_lane=5, speed = 50):
    
    #Max angle deviation coef depends on the speed of the robot
    max_angle_deviation_two_lines = max_angle_deviation_two_lines * 70 / speed
    max_angle_deviation_one_lane = max_angle_deviation_one_lane * 70 / speed
    """
    Using last steering angle to stabilize the steering angle
    This can be improved to use last N angles, etc
    if new angle is too different from current angle, only turn by max_angle_deviation degrees
    """
    if num_of_lane_lines == 2 :
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    else :
        # if only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane
    
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    logging.info('Proposed angle: %s, stabilized angle: %s' % (new_steering_angle, stabilized_steering_angle))
    return stabilized_steering_angle



def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right 
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)





############################
# Frame processing steps
############################
def detect_lane(frame):
    logging.debug('detecting lane lines...')

    edges = detect_edges(frame)
    show_image('edges', edges)

    cropped_edges = region_of_interest(edges)
    show_image('edges cropped', cropped_edges)

    line_segments = detect_line_segments(cropped_edges)
    line_segment_image = display_lines(frame, line_segments)
    show_image("line segments", line_segment_image)

    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    show_image("lane lines", lane_lines_image)

    return lane_lines, lane_lines_image


def detect_edges(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    show_image("hsv", hsv)
    lower_blue = np.array([30, 40, 0])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    show_image("blue mask", mask)

    # detect edges
    edges = cv2.Canny(mask, 200, 400)

    return edges

def detect_edges_old(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    show_image("hsv", hsv)
    for i in range(16):
        lower_blue = np.array([30, 16 * i, 0])
        upper_blue = np.array([150, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        show_image("blue mask Sat=%s" % (16* i), mask)


    #for i in range(16):
        #lower_blue = np.array([16 * i, 40, 50])
        #upper_blue = np.array([150, 255, 255])
        #mask = cv2.inRange(hsv, lower_blue, upper_blue)
       # show_image("blue mask hue=%s" % (16* i), mask)

        # detect edges
    edges = cv2.Canny(mask, 200, 400)

    return edges


def region_of_interest(canny):
    height, width = canny.shape
    mask = np.zeros_like(canny)

    # only focus bottom half of the screen

    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    show_image("mask", mask)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 13  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=8,
                                    maxLineGap=4)

    if line_segments is not None:
        for line_segment in line_segments:
            logging.debug('detected line_segment:')
            logging.debug("%s of length %s" % (line_segment, length_of_line_segment(line_segment[0])))

    return line_segments


leftLineAge = 0
rightLineAge = 0

def average_slope_intercept(frame, line_segments):
    global leftLineAge
    global rightLineAge
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        leftLineAge += 1
        if leftLineAge > 1:
            lane_lines.append(make_points(frame, left_fit_average))
    else:
        leftLineAge = 0

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        rightLineAge += 1
        if rightLineAge > 1:
            lane_lines.append(make_points(frame, right_fit_average))
    else:
        rightLineAge = 0
    print("----------------")
    print("-> right age:  ", rightLineAge)
    print("<- left age:  ", leftLineAge)
    logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]
    print(lane_lines)
    return lane_lines


def make_points(frame, line):
    global showSlope
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    if showSlope:
        print("slope:  ", slope)
    x1 = max(-width, min(2 * width, int((y1 - intercept) / (slope+                        0.0000001))))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / (slope+0.0000001))))
    return [[x1, y1, x2, y2]]


def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)



###########################
# Utility Functions
############################
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=10):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right 
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image