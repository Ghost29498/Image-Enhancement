import SRNDeblur.models.model as model
import easydict
import cv2

"""
initializes the model
    Parameters:
        modelPath: 'path to the model'
    Returns:
        returns the model
"""
def init(modelPath):
    #args ->dictionary to run the model
    #   phase -> 'test' model
    #   modelPath -> path to model
    args = easydict.EasyDict({
        "phase": 'test',
        "modelPath":modelPath
    })
    deblur = model.DEBLUR(args)
    return(deblur)
"""
removes the blur from the image
    Parameters:
        imagePath: 'path to image'
        deblur: 'model'
    Returns:
        returns the clear image
"""
def removeBlur(imagePath,deblur):
    blur = cv2.imread(imagePath)
    clear = deblur.removeBlur(blur)
    return(clear)
    
if __name__ == '__main__':
    imagePath = 'C:/Users/ShriramSubramanian/Motion Deblur/SRNDeblur/testing_set/08.png'
    modelPath= 'G:\\Print out\\writeup\\demoOutput\\srndeblur_models\\color'
    deblur = init(modelPath)
    clear = removeBlur(imagePath,deblur)
    cv2.imwrite('C:/Users/ShriramSubramanian/Motion Deblur/SRNDeblur/output/clearImage23.png',clear)