import SRNDeblur.models.model as model
import easydict
import cv2

class MotionDeBlur:
    def __init__(self,modelPath):
        """
        initializes the model
        Parameters:
            modelPath: 'path to the model'
        Returns:
            returns the model
        Model Parameters
        args ->dictionary to run the model
           phase -> 'test' model
           modelPath -> path to model
        """
        args = easydict.EasyDict({
            "phase": 'test',
            "modelPath":modelPath
        })
        deblur = model.DEBLUR(args)
        self.model = deblur
    def removeBlur(self,imagePath):
        """
        removes the blur from the image
        Parameters:
            imagePath: 'path to image'
            deblur: 'model'
        Returns:
            returns the clear image
        """
        blur = cv2.imread(imagePath)
        clear = self.model.removeBlur(blur)
        return(clear)

if __name__=='__main__':
    imagePath = 'C:/Users/ShriramSubramanian/Motion Deblur/SRNDeblur/testing_set/08.png'
    modelPath= 'G:/Print out/writeup/demoOutput/srndeblur_models/color/'
    deblur = MotionDeBlur(modelPath)
    clear = deblur.removeBlur(imagePath)
    cv2.imwrite('C:/Users/ShriramSubramanian/Motion Deblur/SRNDeblur/output/clearImage231.png',clear)