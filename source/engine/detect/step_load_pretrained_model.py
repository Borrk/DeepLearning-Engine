from engine.steps.IStep import IStep
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import pickle

from keras.applications import vgg16
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.nasnet import preprocess_input, NASNetMobile
from keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2

class step_load_pretrained_model(IStep):
    """ load trained model """

    def __init__(self, output_channel, name=None ):
        super().__init__(self, output_channel, name)

    def IRun(self):
        # load the trained convolutional neural network and the label binarizer
        print("[INFO] loading network...")
        model = self.load_model(self.model_name, self.weights )
        
        lb = pickle.loads(open(self.labels, "rb").read())
        print(lb.classes_)
        
        self.output_channel['model'] = model
        self.output_channel['labels'] = lb
        self.output_channel['image_size'] = self.image_size


    def IParseConfig( self, config_json ):
        self.model_name = config_json['model_name']
        self.labels  = config_json['labels']
        self.weights  = config_json['weights']
        self.image_size = (config_json['image_size'][0], config_json['image_size'][1])

    def IDispose( self ):
        pass

    def load_model( model_name, weights ):
        if model_name == 'vgg16':
            model = vgg16.VGG16(weights=weights, include_top=True)
        elif model_name == 'MobileNet':
            model = MobileNet(include_top=True, weights=weights)
        elif model_name == 'MobileNetV2':
            model = MobileNetV2(include_top=True,  weights=weights)
        elif model_name == 'NASNetMobile':
            model = NASNetMobile(include_top=True,  weights=weights)
        else:
            raise Exception( "Not supported model type" )

        return model