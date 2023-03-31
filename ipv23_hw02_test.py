# Import the required libraries for image processing

import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

# Define the directory
dir = '/IPV23_HW2/test_imgs/' # File path

# Define the functions for the load and save the input image

def loadImg(in_fname):
  img = cv2.imread(dir + in_fname)

  if img is None:
    print('Image load failed!')
    sys.exit()

  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.title("Input RGB Image")
  plt.show()

  return img  

## Save image file
def saveImg(out_img, out_fname):
  cv2.imwrite(dir + out_fname, out_img)

  ##############################################
#            Convolution Operator            #
##############################################
def conv(image, filter):
  
  if len(image.shape) == 3:    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    

  ih, iw = image.shape    # width, height of image
  fh, fw = filter.shape   # width, height of filter

  # plt.imshow(image, cmap='gray')
  # plt.show()

  output = np.zeros(image.shape)

  pad_h = int((fh - 1) / 2)
  pad_w = int((fw - 1) / 2)

  padded_image = np.zeros((ih + (2 * pad_h), iw + (2 * pad_w)))
  padded_image[pad_h:padded_image.shape[0] - pad_h, pad_w:padded_image.shape[1] - pad_w] = image

  # plt.imshow(padded_image, cmap='gray')
  # plt.show()

  for i in range(ih):  # Vertical positions
    for j in range(iw):  # Horizontal positions
      output[i, j] = np.sum(filter * padded_image[i:i+fh, j:j+fw])
  
  # plt.imshow(output, cmap='gray')
  # plt.show()

  return output


##############################################
#                Sobel Filter                #
##############################################
def edgeSobel_x():

  ############################################
  ##                                        ##
  ##             YOUR CODE HERE             ##
  ##                                        ##
  ############################################

  return output_filter


def edgeSobel_y():

  ############################################
  ##                                        ##
  ##             YOUR CODE HERE             ##
  ##                                        ##
  ############################################

  return output_filter
