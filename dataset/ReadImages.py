
# coding: utf-8

# In[16]:

import glob
import os.path as path
from PIL import Image
import torchvision.transforms as transforms


# In[8]:

def readImagesInCLass(folder='.'):
    """
        Read a folder containing images with the structure :
            folder
                --class1
                    --image1
                    --image2
                --class2
                    --image3
                    --image3
        
        Return :
            list of couple : (image, class)
    """
    
    exts = ('*.jpg', '*.JPG', '*.JPEG', "*.png")
    r = []
    for el in glob.iglob(path.join(folder, '*')):
        if path.isdir(el):
            for ext in exts:
                r.extend( [(im, el.split('/')[-1]) for im in glob.iglob(path.join(el, ext)) ] )
    return r


# In[12]:

def readImageswithPattern(folder='.', matchFunc=lambda x:x.split('.')[0]):
    """
        Read a folder containing images where the name of the class is in the filename
        the match function should return the class given the filename
        Return :
            list of couple : (image, class)
    """
    exts = ('*.jpg', '*.JPG', '*.JPEG', "*.png")
    r = []
    for ext in exts:
        r.extend( [(im, matchFunc(im)) for im in glob.iglob(path.join(folder, ext)) ] )
    return r


# In[20]:

def openAll(imageList, size=0 ):
    if size == 0:
        return [Image.open(im) for im, c in imageList]
    else:
        return [Image.open(im).resize(size) for im, c in imageList]
        


# In[ ]:



