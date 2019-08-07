from picamera.array import PiRGBArray
from picamera import PiCamera
import os
from time import sleep

class camera:
    options = {}

    def __init__(self ):
        self.options['framerate'] = 20
        self.options['resolution'] = (320,240)
        self.options['resize'] = None
        self.options['use_video_port'] = True

        self.target = PiCamera()
        self.target.resolution = self.options['resolution']
        self.target.framerate = self.options['framerate']
        sleep(1)

    def config(self, options ):
        for key in options.keys():
            self.options[key] = options[key]

        self.target.resolution = self.options['resolution']
        self.target.framerate = self.options['framerate']

        sleep(1)


    def capture(self):
        with PiRGBArray(self.target) as output:
            # allow the camera to warmup
            sleep(0.1)
 
            # grab an image from the camera
            self.target.capture(output, format="rgb", use_video_port=self.options['use_video_port'] )
            image = output.array
            return image
        
    def capture_continuous(self, callback ):
        if callback == None:
            raise Exception( "No callback function." )

        with PiRGBArray(self.target) as output:
            # allow the camera to warmup
            sleep(0.1)
 
            # grab an image from the camera
            for foo in self.target.capture_continuous(output, format="rgb", use_video_port=self.options['use_video_port'] ):            
                # process by callback
                if callback( output.array ) == True: # stop
                    break
            
                output.truncate(0)

    def record_video( self, seconds, videofile, resolution = None ):
        """ record video
        supported format:
        'h264' - Write an H.264 video stream
        'mjpeg' - Write an M-JPEG video stream
        'yuv' - Write the raw video data to a file in YUV420 format
        'rgb' - Write the raw video data to a file in 24-bit RGB format
        'rgba' - Write the raw video data to a file in 32-bit RGBA format
        'bgr' - Write the raw video data to a file in 24-bit BGR format
        'bgra' - Write the raw video data to a file in 32-bit BGRA format

        """
        
        if resolution != None:
            self.target.resolution = resolution

        self.target.start_preview()
        self.target.start_recording( videofile )

        self.target.wait_recording( seconds )
        self.target.stop_recording()
