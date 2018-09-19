#!/usr/bin/python
# map the trainIds to 1,2,3,4,5,6,7......
from __future__ import print_function
import os, glob, sys
import PIL.Image as Image
import numpy as np

# The main method
def main():
    # Where to look for Cityscapes
#    cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'datasets/cityscapes')
    cityscapesPath = os.path.dirname(os.path.realpath(__file__))
    print("cityscapesPath:",cityscapesPath) 
    # how to search for all ground truth
    searchTrain   = os.path.join( cityscapesPath , "gtFine/train"  , "*" , "*_gt*_instanceTrainIds.png" )
    searchVal = os.path.join( cityscapesPath , "gtFine/val" , "*" , "*_gt*_instanceTrainIds.png" )
    searchTest = os.path.join( cityscapesPath , "gtFine/test" , "*" , "*_gt*_instanceTrainIds.png" )

    # search files
    filesTrain = glob.glob( searchTrain )
    filesTrain.sort()
    filesVal = glob.glob( searchVal )
    filesVal.sort()
    filesTest = glob.glob( searchTest )
    filesTest.sort()

    # concatenate fine and coarse
    files = filesTrain + filesVal + filesTest# use this line if fine is enough for now.

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files:
        
        # create the output filename
        dst = f.replace( "_instanceTrainIds.png" , "_instanceLabelmeIds_align.png" )
        if not os.path.exists(dst):
            # do the conversion
            labelImg = Image.open(f)
            labelNP = np.asarray(labelImg)
            labelUnique = np.unique(labelNP)
            numLabel = len(labelUnique)
            
            objectLabel = []
            for n in range(numLabel):
                
    #            if labelUnique[n] == 11 or labelUnique[n] == 12 or labelUnique[n] == 13 or labelUnique[n] == 14 or labelUnique[n] == 15 or labelUnique[n] == 17 or labelUnique[n] == 18:
                if (labelUnique[n] >= 11000 and labelUnique[n] < 16000) or (labelUnique[n] >= 17000 and labelUnique[n] < 19000):
                    objectLabel.append(labelUnique[n])
            
            changeLabel = range(1, len(objectLabel)+1)
            
            print("objectLabel:", objectLabel)        
            print("changeLabel:", changeLabel)
            
            width, height = labelImg.size
            
            changeLabelNp = np.zeros([height, width], dtype=np.uint8)
    #        size = (width, height)
    #        changeLabelImg = Image.new("L", size, 0)
    #        changeLabelImg.load()
    #    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    #    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    #    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    #    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    #    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    #    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    #    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    
            for i in range(width):
                for j in range(height):
                    
                    at_pixel = labelImg.getpixel((i, j))
                    
                    for k in range(len(objectLabel)):
                        if at_pixel == objectLabel[k]:
                            changeLabelNp[j, i]  = changeLabel[k]
                            
    #                if at_pixel ==  11:
    #                    changeLabelImg[i, j] = 1
    #                elif at_pixel ==  12:
    #                    changeLabelImg[i, j] = 2
    #                elif at_pixel ==  13:
    #                    changeLabelImg[i, j] = 3 
    #                elif at_pixel ==  14:
    #                    changeLabelImg[i, j] = 4 
    #                elif at_pixel ==  15:
    #                    changeLabelImg[i, j] = 5  
    #                elif at_pixel ==  17:
    #                    changeLabelImg[i, j] = 6 
    #                elif at_pixel ==  18:
    #                    changeLabelImg[i, j] = 7
                        
    #                if at_pixel ==  11:
    #                    changeLabelNp[j, i] = 1
    #                elif at_pixel ==  12:
    #                    changeLabelNp[j, i] = 2
    #                elif at_pixel ==  13:
    #                    changeLabelNp[j, i] = 3 
    #                elif at_pixel ==  14:
    #                    changeLabelNp[j, i] = 4 
    #                elif at_pixel ==  15:
    #                    changeLabelNp[j, i] = 5  
    #                elif at_pixel ==  17:
    #                    changeLabelNp[j, i] = 6 
    #                elif at_pixel ==  18:
    #                    changeLabelNp[j, i] = 7 
            # create the output filename
            dst = f.replace( "_instanceTrainIds.png" , "_instanceLabelmeIds_align.png" )
    #        print("dst:",dst)                  
            changeLabelImg = Image.fromarray(changeLabelNp)
            changeLabelImg.save(dst)
    
            # status
            progress += 1
            print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
            sys.stdout.flush()


# call the main
if __name__ == "__main__":
    main()