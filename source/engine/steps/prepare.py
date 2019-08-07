from engine.steps.IStep import IStep
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os

class step_prepare(IStep):
    """Calculate or parse the config file to determine the count of classess"""
    def __init__(self, output_channel, name=None ):
        super().__init__(self, output_channel, name)

    def IRun(self):
        # binarize the labels
        lb = LabelBinarizer()
        encoded_labels = np.array(self.labels)
        encoded_labels = lb.fit_transform(encoded_labels)

        self.output_channel['n_classes'] = self.n_classes
        self.output_channel['train_path'] = self.train_path
        self.output_channel['labels'] = self.labels
        self.output_channel['encoded_labels'] = encoded_labels
        self.output_channel['labelBinarizer'] = lb

    def IParseConfig( self, config_json ):
        self.train_path = config_json['train_path']
        if 'test_path' in config_json:
            self.test_path = config_json['test_path']
        if 'test_split_ratio' in config_json:
            self.test_split_ratio = config_json['test_split_ratio']

        self.labels = self.pick_labels( self, config_json['labels'] )
        self.n_classes = len(self.labels)
        
        #self.outputObjects['n_classes']=self.n_classes

    def IDispose( self ):
        pass

    def pick_labels( self, label_source ):
        if label_source == "extract_from_train_data_path":
            from utils.data_preparer_splitdataset import data_preparer_splitdataset
            labels = data_preparer_splitdataset.pick_labels_from_folders( self.train_path )
        elif isinstance(label_source, list):
            labels = label_source
        elif os.path.isfile( label_source): # read from label file
            labels = data_preparer_splitdataset.pick_labels_from_file( label_source )
        else:
            raise Exception( "Unknown label source" )

        return labels # add later