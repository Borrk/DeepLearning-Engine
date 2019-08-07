from engine.steps.IStep import IStep
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from keras.applications import vgg16
from keras.applications import vgg19


from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer

class build_pretrained_model(IStep):
    """build pre trained model"""

    create_model_func = None

    def __init__(self, output_channel, name,  create_model_func ):
        super().__init__(self, output_channel, name)
        self.create_model_func = create_model_func

    def IRun(self):
        if self.create_model_func == None:
            raise Exception( "No create model function!" )

        model, preprocess, imgsize = self.create_base_model(self)
        self.output_channel['model'] = model
        self.output_channel['preprocess_input'] = preprocess
        self.output_channel['image_size'] = imgsize


    def IParseConfig( self, config_json ):
        self.weights = config_json['weights']
        self.include_top = config_json['include_top']
        self.train_conv_layers = config_json['train_conv_layers']

    def IDispose( self ):
        pass

    def create_base_model( self ):
        # create the base pre-trained model
        base_model, preprocess, imgsize = self.create_model_func( weights=self.weights, include_top=self.include_top )
        
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
       
        # let's add a fully-connected layer
        n_classes = self.output_channel['n_classes']
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 3 classes
        predictions = Dense(n_classes, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional layers
        if self.train_conv_layers == False:
            for layer in base_model.layers:
                layer.trainable = False

        return model, preprocess, imgsize



class build_model_vgg16(build_pretrained_model):
    """build model vgg16"""

    def __init__(self, output_channel, name=None ):
        super().__init__(self, output_channel, name, self.create_vgg16 )

    def create_vgg16( weights, include_top ):        
        """ create vgg16 base model """

        base_model = vgg16.VGG16(weights=weights, include_top=include_top)
        imagesize=(224,224)

        return base_model, vgg16.preprocess_input, imagesize

class build_model_vgg19(build_pretrained_model):
    """build model vgg16"""

    def __init__(self, output_channel, name=None ):
        super().__init__(self, output_channel, name, self.create_vgg19 )

    def create_vgg19( weights, include_top ):        
        """ create vgg19 base model """

        base_model = vgg19.VGG19(weights=weights, include_top=include_top)
        imagesize=(224,224)

        return base_model, vgg19.preprocess_input, imagesize