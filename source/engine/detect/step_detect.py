from engine.steps.IStep import IStep
from keras.models import load_model
from keras.preprocessing.image import img_to_array
#from sklearn.preprocessing import LabelBinarizer
import numpy as np
import time
import cv2

from imutils.video import VideoStream
import imutils

class step_detect(IStep):
    """ detect driving direction """

    def __init__(self, output_channel, name=None ):
        super().__init__(self, output_channel, name)

    def IRun(self):
        # load the trained convolutional neural network and the label binarizer
        print("[INFO] loading network...")
        self.model = self.output_channel['model']
        self.labels  = self.output_channel['labels']
        self.image_size = self.output_channel['image_size']
        #self.resolution = self.output_channel['resolution']        

        camera = self.output_channel['camera']
        self.detect( model=self.model, lb=self.labels, resize=self.image_size, camera=camera )


    def IParseConfig( self, config_json ):
        pass

    def IDispose( self ):
        pass
    
    @staticmethod
    def detect(model, lb, resize, camera ):
        print( "[I] start detecting...")  
    
        #camera.start()

        i=1
        while True:
            t0 = time.time()
            img = camera.read()
            #img = imutils.resize(img, width=w, height=h)           
            t1 = time.time()

            print( "[I] capture image  " + str(i)+': ' + str(t1-t0) )
                        
            image = cv2.resize(img, resize )
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
    
            t2 = time.time()
            label='haha'
            print( " [C] predicting..." + str(t2-t1) )
            #"""   
            proba = model.predict(image)[0]
            idx = np.argmax(proba)
            #label = lb.classes_[idx].lower()
            t3 = time.time()
            print( "[C] label=" + label + ': ' + str(t3-t2))
            #"""
        
            cv2.putText(img, label, ( 10, 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 2 )
            cv2.imshow("img", img)
            
            #f=str(i)+".jpg"
            #cv2.imwrite(f, img )

            i=i+1

            key = cv2.waitKey(1)&0xFF
            if key == ord('q'):
                print( "[I] Quit..." )
                break

