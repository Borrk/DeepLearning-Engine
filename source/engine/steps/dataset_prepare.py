from sklearn.preprocessing import LabelBinarizer

from engine.steps.IStep import IStep
from utils.data_preparer_splitdataset import data_preparer_splitdataset

class dataset_prepare_one_folder(IStep):
    """ base class for dataset preparing """

    def __init__(self, output_channel, name=None ):
        super().__init__(self, output_channel, name)

    def IRun(self):
        data_path = self.output_channel['train_path']
        size = self.output_channel['image_size']
        preprocess_input = self.output_channel['preprocess_input']

        loader = data_preparer_splitdataset( data_path, self.test_split_ratio, 0.0 )
        (trainX, trainY, testX, testY, validX, validY, lb) = loader.prepare_data( size, preprocess=preprocess_input )
       
        self.output_channel['train_data'] = [trainX, trainY]
        self.output_channel['test_data'] = [testX, testY]


    def IParseConfig( self, config_json ):
        self.test_split_ratio = config_json['test_split_ratio']
        if not 0.0 < self.test_split_ratio < 1.0:
            raise Exception( "test split ratio is out of range:" + str(self.test_split_ratio) )
    
    def IDispose( self ):
        pass
