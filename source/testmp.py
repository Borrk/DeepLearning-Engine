# USAGE
# python classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
from imutils import paths
import pickle
import cv2

#from imutils.video import VideoStream

import os
import multiprocessing
import time

#from keras.applications.mobilenet import MobileNet, preprocess_input

#temp
imagePaths = sorted(list(paths.list_images('E:\\PGD\\DataMining\\Tools\\flower_photos-org')))
for file in imagePaths:
    print( file )
    label = file.split(os.path.sep)#[-2]
    print(label)

'''
def ExecClassify( imagefile, model, labels, IMAGE_DIMS ):
    image = cv2.imread(imagefile)
    #output = image.copy()
 
    # pre-process the image for classification
    image = cv2.resize(image, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)


    # classify the input image
    #print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx].lower()

    # we'll mark our prediction as "correct" of the input image filename
    # contains the predicted label text (obviously this makes the
    # assumption that you have named your testing image files this way)
    filename = imagefile[imagefile.rfind(os.path.sep) + 1:].lower()
    correct = "correct" if filename.rfind(label) != -1 else "incorrect"

    return correct, proba[idx] * 100, label, filename

IMAGE_DIMS = (224, 224, 3)

def get_image( queue, imagePaths ):
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)

        f= imagePath.split(os.path.sep)[-1]
        print( "[I] put: " + f )
        queue.put(image )
        #time.sleep(1)

def classify( queue, model, lb ):
    i=0
    print( "Classify started: ".format("{:d}"), os.getpid() )
    while True:
        t0 = time.time()
        try:
            img = queue.get()
        
            # pre-process the image for classification
            image = cv2.resize(img, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

        except Exception as e:
            print( " [C] queue empty" )
            time.sleep(0.5)
        else:
            #"""   
            proba = model.predict(image)[0]
            idx = np.argmax(proba)
            label = lb.classes_[idx].lower()
          
            print( "[C] label=" + label)
            #"""
            cv2.putText(img, label, ( 10, 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 2 )
            #cv2.imshow("img", img)
            f=str(i)+".jpg"
            i=i+1
            cv2.imwrite(f, img )


args = { "model": "E:\PGD\DataMining\Tools\AddLabel\TransferLearn\pokedex_VGG16",
          "labelbin": "E:\PGD\DataMining\Tools\AddLabel\TransferLearn\lb.pikle",
          "image": "E:\PGD\DataMining\Preprocess\ProcessedData\BlackFilled_W320_H240\Camillia"
}


IMAGE_DIMS = (224, 224, 3)

if __name__ == '__main__':
    queue = multiprocessing.Queue(6)

    # load the trained convolutional neural network and the label
    # binarizer
    print("[INFO] loading network...")
    model = load_model(args["model"])
    lb = pickle.loads(open(args["labelbin"], "rb").read())

    # load the image
    results=[]
    imagePaths = sorted(list(paths.list_images(args["image"])))

    p1 = multiprocessing.Process(target=get_image,args=(queue, imagePaths))
    p2 = multiprocessing.Process(target=classify,args=(queue, model, lb))
    p2.start()
    p1.start()

    queue.close()
    p1.join()
    p2.join()

'''




"""
# try to classify
valid = 0
invalid = 0
for imagePath in imagePaths:
    result = ExecClassify( imagePath, model, lb, IMAGE_DIMS )
    results.append(result)
    if ( result[0] == "correct" ):
        valid+=1
        if ( result[1] <= 0.7):
            print(result)
    else:
        invalid = invalid+1
        print( result)

print( "Valid count={:d}", valid)
print( "Invalid count={:d}", invalid)
print( "Acc={:.1f}", 100.0*valid/(valid+invalid))

"""
