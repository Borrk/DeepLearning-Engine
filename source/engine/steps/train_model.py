from engine.steps.IStep import IStep
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import numpy as np
import pickle

class train_model(IStep):
    """train model"""

    callbackTags = []
    callback_list = []

    # function for preparing training parameters, like callback, ImageDataGenerator
    setup_model_training_parameters_func = None
    # training function, like model.fit_generator()
    model_training_func = None 
    # save training results(model, weights, lables, etc)
    post_training_func = None  

    def __init__(self, output_channel, name=None):
        super().__init__(self, output_channel, name)
        self.setup_model_training_parameters_func = self.setup_model_training_parameters
        self.post_training_func = self.post_process

    def IRun(self):
        if self.model_training_func == None:        
            self.output_channel['error_msg'] = "No model training function defined in " + self
            self.output_channel['error_type'] = "fatal"
            raise Exception( "No create model function!" )
        model = self.output_channel['model']
        if model == None:        
            self.output_channel['error_msg'] = "No model created in previous step."
            self.output_channel['error_type'] = "fatal"
            raise Exception( "No model created!" )

        # step 1: setup parameters
        if self.setup_model_training_parameters_func != None:
            self.setup_model_training_parameters_func(self)

        # step 2: train the model
        self.model_training_func(self, model)
        
        # step 3: post processing, like save model, weights, labels, etc.
        if self.post_training_func != None:
            self.post_training_func(self, model)


    def IParseConfig( self, config_json ):
        self.augment_data = config_json['augment_data']
        self.callbackTags = config_json['callbacks']
        self.dataset_source = config_json['dataset_source']

        if self.dataset_source == "preloaded":
            self.model_training_func = self.train_model_for_getting_data_from_memory
        elif self.dataset_source == "drive":
            self.model_training_func = self.train_model_for_getting_data_from_drive
        else:
            # may be set in derived class
            self.output_channel['error_msg'] = "No model training function type is recognized in " + self
            self.output_channel['error_type'] = "warning"

        self.output_path = config_json['output_path']
        self.model_path = config_json['model_path']
        self.labels_path = config_json['labels_path']

        self.batch_size = self.output_channel['batch_size']
        self.epochs    = self.output_channel['epochs']

    def IDispose( self ):
        pass

    def setup_model_training_parameters( self ):        
        if self.augment_data == True:
            self.generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.01,
	                height_shift_range=0.01, shear_range=0.02, zoom_range=0.2,
	                horizontal_flip=True, fill_mode="nearest")

        for callback in self.callbackTags:
            if callback == 'modelcheckpoint_keepbest':
                from utils.modelcheckpoint_keepbest import modelcheckpoint_keepbest
                checkpoint = modelcheckpoint_keepbest( monitor='val_loss', mode='min', period=1)
                self.callback_list.append(checkpoint)
                self.callback_checkbest = checkpoint

    def train_model_for_getting_data_from_memory(self, model):
        trainX, trainY = self.output_channel['train_data']
        testX, testY = self.output_channel['test_data']

        print("[INFO] training network...")
        print("Len of train data= ", len(trainX))

        his = model.fit_generator(
	        self.generator.flow(trainX, trainY, batch_size=self.batch_size),
	        validation_data=(testX, testY),
	        epochs=self.epochs, verbose=1,
            workers=4,
            callbacks = self.callback_list)

        self.output_channel["training_history"] = his
        self.history = his

    def train_model_for_getting_data_from_drive(self, model):
        print("[INFO] training network...")
        print("Len of train data= ", len(trainX))
        
        raise Exception("Not implemented yet!")

    def post_process(self, model):
        print("[INFO] after training process...")
        self.show_train_hist(self)

        # save the model to disk
        print("[INFO] serializing network...")
        model.save(self.model_path)

        # save the label binarizer to disk
        print("[INFO] serializing label binarizer...")
        lb = self.output_channel['labelBinarizer']

        f = open(self.labels_path, "wb")
        f.write(pickle.dumps(lb))
        f.close()

        #save best model
        if self.callback_checkbest != None:            
            best_path = paths.os.path.join( self.output_path, "best_model")
            bestmodel = self.callback_checkbest.get_best_mode()
            print("[INFO] serializing best network...")
            bestmodel.save(best_path)


    def show_train_hist(self):
        his = self.history

        # plot the training loss and accuracy
        import matplotlib.pyplot as plt

        plt.style.use("ggplot")
        plt.figure()
        N = self.epochs
        plt.plot(np.arange(0, N), his.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), his.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), his.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), his.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        #plt.savefig(args["plot"])
