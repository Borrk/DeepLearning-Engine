from keras.preprocessing import image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import random
import pickle
import os
import os.path

class data_preparer_splitdataset(object):
 
    def __init__( self, data_path, testdata_percent=0.1, validatedata_percent=0.1 ):
        self.data_path = data_path
        self.test_size = testdata_percent
        self.valid_size = validatedata_percent
   

    def  prepare_data(self, size, is_shuffled=True, is_convert2array=True, preprocess=None, colormode='rgb' ):
        data=[]
        labels=[]
        nbytes = 0

        # grab the image paths and randomly shuffle them
        img_paths = sorted(list(paths.list_images( self.data_path )))
    
        if (is_shuffled):
            random.seed(42)
            random.shuffle(img_paths)

        # loop over the input images
        for file in img_paths:
            img = image.load_img( file, target_size=size ) #, color_mode=colormode )
            if True==is_convert2array:
                img = image.img_to_array(img)  
                # works only if is_convert2array==True
                if preprocess != None:
                    img = preprocess(img)

            data.append(img)

	        # extract labels from image path
            label = file.split(os.path.sep)[-2]
            labels.append(label)

        print( "images count:{:,}".format( len(data) ) ) 

        # convert to array
        data = np.array(data, dtype="float")
        print("[INFO] data matrix: {:.2f}MB".format(
	        data.nbytes / (1024 * 1000.0)))

        # binarize the labels
        lb = LabelBinarizer()
        labels = np.array(labels)
        labels = lb.fit_transform(labels)

        testX= testY= validX= validY = []
        # partition the data into training, testing and validation splits
        if (self.test_size + self.valid_size) > 0.0:
            (trainX, testX, trainY, testY) = train_test_split(data,
	            labels, test_size=(self.test_size+self.valid_size), random_state=42)
            
            split_size = self.valid_size/(self.test_size + self.valid_size)
            if split_size > 0.0:
                (testX, validX, testY, validY) = train_test_split(testX,
	                testY, test_size = split_size, random_state=42)

        return trainX, trainY, testX, testY, validX, validY, lb;

    @staticmethod
    def load_img(filePath, size, is_convert2array=True, preprocess=None, colormode='rgb' ):
        orgimg = image.load_img( filePath ) 
        img = image.load_img( filePath, target_size=size ) #, color_mode=colormode )
        if True==is_convert2array:
            img = image.img_to_array(img)  
            # works only if is_convert2array==True
            if preprocess != None:
                img = preprocess(img)
        
        label = filePath.split(os.path.sep)[-2]

        # convert to array
        data = np.array(img, dtype="float")
        return orgimg, data, label

    @staticmethod
    def pick_labels_from_folders( path ):
        if not os.path.isdir(path):
           path = os.path.join( "./", path )
        labels = sorted([o for o in os.listdir(path) if os.path.isdir(os.path.join(path,o))])

        return labels

    @staticmethod
    def pick_labels_from_file( path ):        
        labels=[]

        if os.path.isfile(path):
           raise Exception("Not implemented!") # ADD later
        return labels