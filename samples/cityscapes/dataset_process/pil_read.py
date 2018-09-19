import os, glob, sys
import PIL.Image as Image
import numpy as np
# img = Image.open('/home/dllinux/Mask-RCNN/Mask_RCNN/samples/cityscapes/datasets/cityscapes/gtFine/test/berlin/berlin_000000_000019_gtFine_labelChangeIds.png')
# img=np.array(Image.open('/home/dllinux/Mask-RCNN/Mask_RCNN/samples/cityscapes/datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelChangeIds.png').convert('L'))
# img=np.array(Image.open('/home/dllinux/Mask-RCNN/Mask_RCNN/samples/cityscapes/datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png').convert('L'))
imgLabel=np.array(Image.open('/home/dllinux/datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png').convert('L'))
imgLabelTrain=np.array(Image.open('/home/dllinux/datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png').convert('L'))

imgInstance=np.array(Image.open('/home/dllinux/datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_instanceIds.png').convert('L'))
imgInstanceTrain=np.array(Image.open('/home/dllinux/datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_instanceTrainIds.png').convert('L'))

imgChange=np.array(Image.open('/home/dllinux/datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelChangeIds.png').convert('L'))
imgLabelme=np.array(Image.open('/home/dllinux/datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelmeIds.png').convert('L'))

imgInstancelabelme=np.array(Image.open('/home/dllinux/datasets/cityscapes_TrainVal/hanover_000000_005732_gtFine_instanceLabelmeIds_align.png').convert('L'))

numLabel=np.unique(imgLabel)
print('numLabel:',numLabel)

numLabelTrain=np.unique(imgLabelTrain)
print('numLabelTrain:',numLabelTrain)

numInstance=np.unique(imgInstance)
print('numInstance:',numInstance)

numInstanceTrain=np.unique(imgInstanceTrain)
print('numInstanceTrain:',numInstanceTrain)

numChange=np.unique(imgChange)
print('numChange:',numChange)

numLabelme=np.unique(imgLabelme)
print('numLabelme:',numLabelme)

numInstancelabelme=np.unique(imgInstancelabelme)
print('numInstancelabelme:',numInstancelabelme)

# width, height = img.size
# for i in range(width):
#     for j in range(height):
#         at_pixel = img.getpixel((i, j))


#         if at_pixel != 0
#         print(at_pixel)

