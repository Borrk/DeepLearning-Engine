from PIL import Image
from keras.preprocessing import image
from data_preparer_splitdataset import data_preparer_splitdataset
 
def flip_image(image_path):
    """
    Flip or mirror the image
 
    @param image_path: The path to the image to edit
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    #rotated_image.save(saved_location)
    rotated_image.show()
    
    return rotated_image

def flip(filePath):
    img = image.load_img( filePath ) #, color_mode=colormode )
        
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.show()
        
file = 'G:/CarAssist/img/data/left/left_2019-03-25-04_20_20.050000.jpg'
#flip(file)
img, x, label = data_preparer_splitdataset.load_img( file, (96, 300) )

#flip_image(file)