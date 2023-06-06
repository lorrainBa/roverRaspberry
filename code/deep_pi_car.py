import cv2
import os 
import numpy as np
import logging
import math

import picar
import datetime

from hand_coded_lane_follower import HandCodedLaneFollower
#Get the image


_SHOW_IMAGE = True


class DeepPiCar(object):

    __INITIAL_SPEED = 0
    __SCREEN_WIDTH = 320
    __SCREEN_HEIGHT = 240

    def __init__(self):
        """ Init camera and wheels"""
        logging.info('Creating a DeepPiCar...')

        picar.setup()

        logging.debug('Set up camera')
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(3, self.__SCREEN_WIDTH)
        self.camera.set(4, self.__SCREEN_HEIGHT)

        self.pan_servo = picar.Servo.Servo(1)
        self.pan_servo.offset = -30  # calibrate servo to center
        self.pan_servo.write(90)

        self.tilt_servo = picar.Servo.Servo(2)
        self.tilt_servo.offset = 20  # calibrate servo to center
        self.tilt_servo.write(90)

        logging.debug('Set up back wheels')
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.back_wheels.speed = 0  # Speed Range is 0 (stop) - 100 (fastest)

        logging.debug('Set up front wheels')
        self.front_wheels = picar.front_wheels.Front_Wheels()
        self.front_wheels.turning_offset = 22  # calibrate servo to center
        self.front_wheels.turn(90)  # Steering Range is 45 (left) - 90 (center) - 135 (right)

        logging.debug('Code de test')
        logging.debug('deep picar end to end lane follower')
        logging.debug('________________________________________')
        logging.debug('________________________________________')
        self.lane_follower = HandCodedLaneFollower(self)

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        datestr = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        """self.video_orig = self.create_video_recorder('./data/tmp/car_video%s.avi' % datestr)
        self.video_lane = self.create_video_recorder('./data/tmp/car_video_lane%s.avi' % datestr)
        self.video_objs = self.create_video_recorder('./data/tmp/car_video_objs%s.avi' % datestr)"""

        logging.info('Created a DeepPiCar')
        
    def __enter__(self):
        """ Entering a with statement """
        return self

    def __exit__(self, _type, value, traceback):
        """ Exit a with statement"""
        if traceback is not None:
            # Exception occurred:
            logging.error('Exiting with statement with exception %s' % traceback)

        self.cleanup()

    def cleanup(self):
        """ Reset the hardware"""
        logging.info('Stopping the car, resetting hardware.')
        self.back_wheels.speed = 0
        self.front_wheels.turn(90)
        self.camera.release()
        self.video_orig.release()
        self.video_lane.release()
        self.video_objs.release()
        cv2.destroyAllWindows()


    def drive(self,speed=40):
        """ Main entry point of the car, and put it in drive mode

        Keyword arguments:
        speed -- speed of back wheel, range is 0 (stop) - 100 (fastest)
        """

        logging.info('Starting to drive at speed %s...' % speed)
        self.back_wheels.speed = 0
        i = 0
        while self.camera.isOpened():
            _, image_lane = self.camera.read()
            image_objs = image_lane.copy()
            i += 1
            """self.video_orig.write(image_lane)

            self.video_objs.write(image_objs)"""
            """show_image('Detected Objects', image_objs)"""

            image_lane = self.follow_lane(image_lane)
            """self.video_lane.write(image_lane)"""
            show_image('Lane Lines', image_lane)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cleanup()
                break
            
            
    def follow_lane(self, image):
        image,stop = self.lane_follower.follow_lane(image)
        if stop :
            logging.info('No lane lines detected, nothing to do ################################################.')
            logging.info('No lane lines detected, nothing to do.################################################. fd')
            self.back_wheels.speed = 0
            self.front_wheels.turn(90)
        return image

            
def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)
    
            
def main():
    car = DeepPiCar()
    car.drive(40)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s:%(asctime)s: %(message)s')
    
    main()
