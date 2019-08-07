from engine.steps.IStep import IStep
from imutils.video import VideoStream
import imutils

class step_setup_camera(IStep):
    """ setup camera"""
    options = {}

    def __init__(self, output_channel, name=None ):
        super().__init__(self, output_channel, name)
        self.usePiCamera = True

    def IRun(self):
        self.camera=VideoStream(usePiCamera=self.usePiCamera, 
                                resolution=self.options['resolution'], framerate= self.options['framerate'] ).start()
        self.output_channel['camera'] = self.camera
        self.output_channel['resolution'] = self.options['resolution']

    def IParseConfig( self, config_json ):             
        self.options['framerate']       = config_json['framerate']
        self.options['resolution']      = (config_json['resolution'][0],config_json['resolution'][1])
        self.options['usePiCamera']     = config_json['usePiCamera']

    def IDispose( self ):
        try:
            self.target.stop()
        except Exception as e:
            print("close camera exception")