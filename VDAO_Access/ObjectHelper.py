import cv2
import sys
import numpy as np
import os
import warnings
import random
import utils
# To get video information
class ObjectDatabase:
    """
	Class representing any object database (Ex: ALOI or shoesV2). Methods presented here are tools to
    access and work with those images.

        Developed by: Rafael Padilla
        SMT - Signal Multimedia and Telecommunications Lab
        COPPE - Universidade Federal do Rio de Janeiro
        Last modification: Dec 9th 2017 
    """

    def __init__(self, imagesPath, masksPath, extensionImages='png', extensionMasks='png'):
        self.imagesPath = None
        self.masksPath = None
        self.numberOfImageFolders = None
        self.numberOfMaskFolders = None
        self.numberOfImages = None
        self.numberOfMasks = None

        if os.path.isdir(imagesPath):
            self.extenstionImages = extensionImages
            self.imagesPath = imagesPath
            # self.numberOfImageFolders = sum([len(d) for _, d, _ in os.walk(imagesPath)])
            # TODO: Otimizar
            # self.numberOfImages = sum([len(f) for r, d, f in os.walk(imagesPath)])
            # self.numberOfImages = self._getFilesFromFolder(imagesPath, "."+self.extenstionImages)

        if os.path.isdir(masksPath):
            self.extensionMasks = extensionMasks
            self.masksPath = masksPath
            # self.numberOfMaskFolders = sum([len(d) for r, d, f in os.walk(masksPath)])
            # TODO>: Otimizar            
            # self.numberOfMasks = sum([len(f) for r, d, f in os.walk(masksPath)])
            # self.numberOfMasks = self._getFilesFromFolder(masksPath, "."+self.extensionMasks)
        
        # if self.imagesPath != None and self.masksPath != None:
        #     if self.numberOfImageFolders != self.numberOfMaskFolders:
        #         warnings.warn("Number of image classes does not match the number of mask classes (%d and %d) respectively")

    def _getFilesFromFolder(self,folder,extension):
        count = 0
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if os.path.isfile(path) and path.endswith(extension):
                count += 1
            elif os.path.isdir(path):
                count += self._getFilesFromFolder(path, extension)
        return count

    # It looks in the object folder for a random image and its random mask
    def getRandomObject(self):
        allObjects = utils.getAllFilesRecursively(self.imagesPath, self.extenstionImages)
        idx = random.randint(0, len(allObjects))
        objPath = allObjects[idx-1]
        print(objPath)
        # Try to get mask path based on the path of the object
        maskPath = objPath.replace(self.imagesPath,self.masksPath)
        print(maskPath)
        if os.path.isfile(maskPath) == False:
            raise IOError("Not able to get mask of the object %s", objPath)
        # Merge image and the mask
        mergedImage = ObjectDatabase.blendImageAndMask(objPath, maskPath)
        # Get bounding box of the merged image
        (min_x, min_y, max_x, max_y) = self.getBoundingBoxMask(cv2.imread(maskPath))
        # Return merged image and its bounding box position
        return [mergedImage, (min_x, min_y, max_x, max_y)]

    @staticmethod
    def blendImageAndMask(objPath, maskPath):
        # Load images
        img = cv2.imread(objPath)
        mask = cv2.imread(maskPath)
        # Multiply mask by image so we have the object only
        mergedImage = np.multiply(img,mask/255)
        mergedImage = mergedImage.astype(np.uint8)
        return mergedImage

    # Using my method
    @staticmethod
    def blendImageAndBackground(image, mask, background, xIni=0, yIni=0, scaleFactor=1, rotAngle=0, flipHor=False, iteracao1=10, iteracao2=5):
        # Flip horizontally
        if flipHor==True:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        # Rotate counter-clockwise by the angle (in degrees)
        w, h, channels = image.shape
        rotMatrix = cv2.getRotationMatrix2D((w/2,h/2), rotAngle, 1)
        image = cv2.warpAffine(image,rotMatrix,(w,h))
        mask = cv2.warpAffine(mask,rotMatrix,(w,h))
        # Rescale image and mask
        image = cv2.resize(image, None, fx=scaleFactor, fy=scaleFactor,  interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, None, fx=scaleFactor, fy=scaleFactor,  interpolation=cv2.INTER_CUBIC)
        mask = mask[:,:,0] # Get only one channel of the mask
        # It's necessary to get width and height once the image was rescaled
        w, h, _ = image.shape
        # Get mask with smooth inner border representing how much the background will be present in the image
        se = cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(3,3))
        bin_mask = mask/255
        inv_mask = 255-mask
        inv_mask_erode_half = cv2.erode(src=mask, kernel=se, iterations=iteracao2)
        mask_erode_half = 255-inv_mask_erode_half
        inv_mask_erode_full = cv2.erode(src=mask, kernel=se, iterations=iteracao1)
        mask_diff_erode = mask-inv_mask_erode_full
        dist_transf = cv2.distanceTransform(src=mask_diff_erode, distanceType=cv2.DIST_L2, maskSize=5)
        dist_transf = (dist_transf/dist_transf.max())*255
        inv_dist_transf = 255-dist_transf
        final_mask = np.multiply((mask_erode_half/255),inv_dist_transf)
        # Multiply each channel of the image by its mask
        for i in range(channels):
            image[:,:,i] = image[:,:,i] * bin_mask
        # Create images with the same size of the background to be blended
        new_background = np.zeros(background.shape, dtype=background.dtype)
        guide = np.ones(background.shape, dtype=float)
        img1 = np.zeros(background.shape, dtype=background.dtype)
        img2 = background
        img1[yIni:h+yIni,xIni:w+xIni,:] = image
        guide[yIni:h+yIni,xIni:w+xIni,0:3] = cv2.merge((final_mask/255, final_mask/255,final_mask/255))
        # Use the linear relation guide*img1 + ((1-guide)*img2)
        new_background = np.multiply(1-guide,img1) + np.multiply(guide, img2)
        # Find bounding box where the object will be inserted
        auxBackground = np.zeros(background.shape, dtype=background.dtype)
        auxBackground[yIni:h+yIni,xIni:w+xIni,0:3] = cv2.merge((mask, mask, mask))
        min_x, min_y, max_x, max_y = ObjectDatabase.getBoundingBoxMask(auxBackground)
        return new_background, [min_x, min_y, max_x, max_y]

    @staticmethod
    # Blend images using Bruno's method
    def blendImageAndBackground_2(image, mask, background, xIni=0, yIni=0, scaleFactor=1, rotAngle=0, flipHor=False, iteracao1=7, iteracao2=4):
        ###MACROS
        REF_IMG_HEIGHT, REF_IMG_WIDTH,_ = background.shape
        STREL_KERNEL = cv2.getStructuringElement(shape=1, ksize=(3,3))
        ### END OF MACROS
        
        # Flip horizontally
        if flipHor==True:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Combine image and its mask
        bin_mask = mask/255
        w, h, channels = image.shape
        img_org = np.zeros((w,h,3))
        img_org = np.multiply(bin_mask, image)
        obj_rows, obj_cols, obj_channels = img_org.shape
        # Rotate counter-clockwise by the angle (in degrees)
        rotMatrix = cv2.getRotationMatrix2D((w/2,h/2), rotAngle, 1)
        img_org = cv2.warpAffine(img_org,rotMatrix,(w,h))
        mask_rot = cv2.warpAffine(mask,rotMatrix,(w,h))
        # Rescale image and mask
        img_rot = cv2.resize(img_org, None, fx=scaleFactor, fy=scaleFactor,  interpolation=cv2.INTER_CUBIC)
        rsz_rows, rsz_cols, rsz_channels = img_rot.shape
        # CREATE MASK FOR OBJECT PLACEMENT
        img_mask = np.zeros((REF_IMG_HEIGHT,REF_IMG_WIDTH,3))
        rnd_nb_row = yIni
        rnd_nb_col = xIni
        img_mask[yIni:rsz_rows+yIni,xIni:rsz_cols+xIni,:] = img_rot
        # BINARIZATION OF THE MASK
        img_mask_bin = np.array(img_mask)
        img_mask_bin [ img_mask_bin > 0 ] = 255
        img_mask_bin = 255 - img_mask_bin
        # PERFORMS DILATATION OF THE MASK TO REMOVE BLACK BORDER
        img_mask_bin = img_mask_bin.astype(np.uint8)
        img_mask_bin = cv2.cvtColor(img_mask_bin, cv2.COLOR_RGB2GRAY)
        tresh_val, img_mask_bin = cv2.threshold(img_mask_bin, 127, 255, cv2.THRESH_BINARY)
        img_mask_bin_dilate = cv2.dilate(src=img_mask_bin, kernel=STREL_KERNEL, iterations=iteracao1)
        tresh_val, img_mask_bin_inv_dilate = cv2.threshold(img_mask_bin_dilate, 127, 255, cv2.THRESH_BINARY_INV)
        img_mask_bin_backg_dilate = cv2.dilate(src=img_mask_bin_dilate, kernel=STREL_KERNEL, iterations=iteracao2)
        img_mask_bin_backg_inv_dilate = cv2.bitwise_not(img_mask_bin_backg_dilate)
        img_mask_bin_inv_dilate_ORG = np.array(img_mask_bin_inv_dilate)
        # TRANSFORMS BINARY IMAGE OF THE DILATATED MASKS INTO RGB
        img_mask_bin_dilate = np.expand_dims(img_mask_bin_dilate,axis=2)
        img_mask_bin_dilate = np.concatenate((img_mask_bin_dilate, img_mask_bin_dilate, img_mask_bin_dilate),axis=2)
        img_mask_bin_inv_dilate = np.expand_dims(img_mask_bin_inv_dilate,axis=2)
        img_mask_bin_inv_dilate = np.concatenate((img_mask_bin_inv_dilate, img_mask_bin_inv_dilate, img_mask_bin_inv_dilate),axis=2)	
        img_mask_bin_backg_dilate = np.expand_dims(img_mask_bin_backg_dilate,axis=2)
        img_mask_bin_backg_dilate = np.concatenate((img_mask_bin_backg_dilate, img_mask_bin_backg_dilate, img_mask_bin_backg_dilate),axis=2)
        img_mask_bin_backg_inv_dilate = np.expand_dims(img_mask_bin_backg_inv_dilate,axis=2)
        img_mask_bin_backg_inv_dilate = np.concatenate((img_mask_bin_backg_inv_dilate, img_mask_bin_backg_inv_dilate, img_mask_bin_backg_inv_dilate),axis=2)
        # GENERATES NEW MASKS
        img_mask_new = np.multiply(img_mask,img_mask_bin_inv_dilate/255)
        img_mask_new_backg = np.multiply(background,img_mask_bin_backg_dilate/255)
        img_mask_bin_dif_dilate = np.subtract(img_mask_bin_inv_dilate,img_mask_bin_backg_inv_dilate)
        # GENERATE BLENDED IMAGE WITHOUT WEIGHTED SUM
        img_ref_cut = np.multiply(background,img_mask_bin_dilate/255)
        img_final = np.add(img_mask_new, img_ref_cut)
        # DISTANCE TRANSFORM FUNCTION FOR THE BORDER OF THE OBJECT
        img_dist_transf = cv2.distanceTransform(src=img_mask_bin_inv_dilate_ORG, distanceType=cv2.DIST_L2, maskSize=5)
        img_dist_transf = np.expand_dims(img_dist_transf,axis=2)
        img_dist_transf = np.concatenate((img_dist_transf, img_dist_transf, img_dist_transf),axis=2)
        img_dist_transf_border = np.multiply(img_dist_transf, img_mask_bin_dif_dilate/255)
        for i in range(3):
            max_val = np.max(np.max(img_dist_transf_border[:,:,i]))
            img_dist_transf_border[:,:,i] = img_dist_transf_border[:,:,i] / max_val
        # GENERATING WEIGHT "IMAGE"
        img_weight_foreg = np.array(img_dist_transf_border)
        img_weight_foreg [img_mask_bin_backg_inv_dilate == 255] = 1
        img_weight_backg = 1 - np.array(img_weight_foreg)
        img_blur_border = np.add(np.multiply(img_weight_foreg,img_mask_new),np.multiply(img_weight_backg,img_mask_new_backg))
         # Find bounding box where the object will be inserted
        min_x, min_y, max_x, max_y = ObjectDatabase.getBoundingBoxMask(img_weight_foreg*255)
        return img_blur_border, [min_x, min_y, max_x, max_y]

    @staticmethod
    def getBoundingBoxMask(mask):
    
        h, w, _ = mask.shape
        min_x = w
        max_x = 0
        # Obs: masks' channels are identical, so checking existence of a pixel with value 255 could be done in any channel
        
        # For every line(row) of the image
        for i in range(h):
            # Get the max position of the pixel whose value is 255
            b = mask[i,:,0][::-1]
            pos_x = len(b) - np.argmax(b)
            if pos_x != w and pos_x > max_x: # checking if pixel of channel 0 is 255
                 max_x = pos_x-1
            # Get the min position of the pixel whose value is 255
            pos_x = np.argmax(mask[i,:,0] == 255) 
            if pos_x != 0 and pos_x < min_x: # checking if pixel of channel 0 is 255
                min_x = pos_x
        
        min_y = h
        max_y = 0
         # For every column of the image
        for i in range(w):
            # Get the max position of the pixel whose value is 255
            b = mask[:,i,0][::-1]
            pos_y = len(b) - np.argmax(b)
            if pos_y != h and pos_y > max_y: # checking if pixel of channel 0 is 255
                 max_y = pos_y-1
            # Get the min position of the pixel whose value is 255
            pos_y = np.argmax(mask[:,i,0] == 255) 
            if pos_y != 0 and pos_y < min_y: # checking if pixel of channel 0 is 255
                min_y = pos_y
        return min_x, min_y, max_x, max_y