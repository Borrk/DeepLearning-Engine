class IStep(object):
    """The interface of engine step, defining the common interfaces(APIs).
    All the specific steps must implement all the interfaces defined in it."""

    name = ''

    def IRun(self):
        raise Exception('derived class must implement this method.')

    def IParseConfig( self, config_json ):
        raise Exception('derived class must implement this method.')
    
    def IDispose( self ):
        raise Exception('derived class must implement this method.')

    def __init__(self, output_channel, name=None ):
        self.output_channel = output_channel
        self.name = name
        self.outputObjects = {}
        self.input_objects = {}

    def set_output_channel(self, output_channel):
        self.output_channel = output_channel

    def get_output_objects(self):
        return self.outputObjects

    def get_input_objects():
        return self.input_objects


