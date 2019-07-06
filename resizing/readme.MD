# Resizing images

The given image is resized to the given size
runResizing.ipynb has an example program on how it works

resizing.py has helper functions to run the code

init(modelPath) -> initialize the model, accepts the modelPath(path to the weights) as parameter, returns the model
resizing(imgname,height,width,model) -> returns the resized images, or returns Boolean False saying the dimention does not fit.
