from engine.steps.IStep import IStep
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import pickle

class step_load_model(IStep):
    """ load trained model """

    def __init__(self, output_channel, name=None ):
        super().__init__(self, output_channel, name)

    def IRun(self):
        # load the trained convolutional neural network and the label binarizer
        print("[INFO] loading network...")
        model = load_model(self.model_file )
        
        lb = pickle.loads(open(self.labels, "rb").read())
        print(lb.classes_)
        
        self.output_channel['model'] = model
        self.output_channel['labels'] = lb
        self.output_channel['image_size'] = self.image_size


    def IParseConfig( self, config_json ):
        self.model_file = config_json['model']
        self.labels  = config_json['labels']
        self.image_size = (config_json['image_size'][0], config_json['image_size'][1])

    def IDispose( self ):
        pass