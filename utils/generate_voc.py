import numpy as np
from glob import glob
import xml.etree.ElementTree as ET
import os
from scipy import misc
import pickle as pkl

class VOC(object):
    
    def __init__(self,path):
        
        try:
            os.stat(path)
        except:
            print 'Path is invalid'
            return None
        
        self.path = os.path.join(os.path.join(path,'Annotations'),'*')
        self.data = None
        self.labels = None
        self.names = []
        annotations = {}
        name = {}
        print 'Extracting annotations from',self.path
        for file_name in glob(self.path):
            tree = ET.parse(file_name)
            root = tree.getroot()
            key = root.find('filename').text
            name[key] = root.find('object').find('name').text
            bndbox = []
            for child in root.find('object').find('bndbox'):
                bndbox.append(child.text)
            annotations[key] = bndbox
        
        self.path = os.path.join(os.path.join(path,'JPEGImages'),'*')
        images = {}
        print 'Extracting images from',self.path
        for file_name in glob(self.path):
            key = file_name.split('/')[-1]
            img = misc.imread(file_name)
            img = misc.imresize(img,size=(28,28))
            images[key] = img
            
        print 'Constructing the Dataset'
        self.data = []
        self.labels = []
        for key in images.keys():
            self.data.append(images[key])
            self.labels.append(annotations[key])
            self.names.append(name[key])
        self.data = np.asarray(self.data)
        self.labels = np.asarray(self.labels)
        
    def __str__(self):
        
        return str((self.data.shape,self.labels.shape))
    
    def save_to_disk(self,path):
        
        ## Pickeling the file and saving it to disk
        with open(os.path.join(path,'data.pkl'),'wb') as fp:
            pkl.dump(self.data,fp)
        
        with open(os.path.join(path,'labels.pkl'),'wb') as fp:
            pkl.dump(self.labels,fp)
            
        with open(os.path.join(path,'name.pkl'),'wb') as fp:
            pkl.dump(self.names,fp)
    
def main():
    path = '../../../Dataset/VOC2007'
    voc = VOC(path)
    print voc
    voc.save_to_disk(path)
    
    
if __name__ == "__main__":
    main()
