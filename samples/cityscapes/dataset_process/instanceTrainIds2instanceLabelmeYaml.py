#!/usr/bin/python
# map the instanceTrainIds to 1,2,3,4,5,6,7......
# one label per instance
from __future__ import print_function
import os, glob, sys
import PIL.Image as Image
import numpy as np
import yaml

# The main method
def main():
    # Where to look for Cityscapes
    cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','..')
    print("cityscapesPath:",cityscapesPath) 
    # how to search for all ground truth
    searchTrain   = os.path.join( cityscapesPath , "gtFine/train"  , "*" , "*_gt*_instanceTrainIds.png" )
    searchVal = os.path.join( cityscapesPath , "gtFine/val" , "*" , "*_gt*_instanceTrainIds.png" )

    # search files
    filesTrain = glob.glob( searchTrain )
    filesTrain.sort()
    filesVal = glob.glob( searchVal )
    filesVal.sort()

    # concatenate fine and coarse
    files = filesTrain + filesVal# use this line if fine is enough for now.

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files:
        # do the conversion
        labelImg = Image.open(f)
        labelNP = np.asarray(labelImg)
        labelUnique = np.unique(labelNP)
        numLabel = len(labelUnique)
        
        instance_names_to_ids = {'BG':0}

        for n in range(numLabel):
            
            if (labelUnique[n] >= 11000 and labelUnique[n] < 16000) or (labelUnique[n] >= 17000 and labelUnique[n] < 19000):

                class_id = labelUnique[n] // 1000
                instance_id = labelUnique[n] % 1000 +1

                if class_id == 11:
                    class_name = 'person'
                elif class_id == 12:
                    class_name = 'rider'
                elif class_id == 13:
                    class_name = 'car'
                elif class_id == 14:
                    class_name = 'truck'                    
                elif class_id == 15:
                    class_name = 'bus'
                elif class_id == 17:
                    class_name = 'motorcycle'
                elif class_id == 18:
                    class_name = 'bicycle'

                instance_label = class_name + str(instance_id)

                instance_names_to_ids.update({instance_label:labelUnique[n]})
        
        # do the conversion
        try:
        
            # label_values must be dense
            label_names = []
            for ln, lv in sorted(instance_names_to_ids.items(), key=lambda x: x[1]):
#                label_values.append(lv)
                label_names.append(ln)            
#                print(ln)

            # if len(label_names) > 1:
            info = dict(label_names=label_names)
            dst = f.replace( "_instanceTrainIds.png" , "_instanceLabelmeNames_align.yaml" )
            with open(dst, 'w') as dst_dir:
                yaml.safe_dump(info, dst_dir, default_flow_style=False)
            # else:
            #     print('empty dst is:', dst)
            
        except:
            print("Failed to convert: {}".format(f))
            raise

        # status
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()


# call the main
if __name__ == "__main__":
    main()