# Import the required libraries for image processing

import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import time

# Define the directory
dir = '/IPV23_HW1/test_imgs/' # File path

# Define the functions for the load and save the input image

def loadImg(in_fname):
  img = cv2.imread(dir + in_fname)

  if img is None:
    print('Image load failed!')
    sys.exit()

  # print(img.shape)
  # cv2.imshow('img', img)

  return img  

## Save image file
def saveImg(out_img, out_fname):
  cv2.imwrite(dir + out_fname, out_img)

##############################################
#            Convolution Operator            #
##############################################
def conv(image, filter):

  ih, iw, id = image.shape  # width, height, dimension of image
  fh, fw = filter.shape     # width, height of filter
  d, i, j = 0, 0, 0

  out = np.zeros((ih-fh+1,iw-fw+1,id))

  for d in range(id): # RGB channel
    for i in range(ih-fh+1):  # Vertical positions
      for j in range(iw-fw+1):  # Horizontal positions
        out[i,j,d] = np.sum(filter*image[i:i+fh, j:j+fw, d])

  if id == 1:
    return np.resize(out,(out.shape[0], out.shape[1])).astype(np.uint8)
  else:
    return out.astype(np.uint8)


##############################################
#                 Mean Filter                #
##############################################
def meanFilter(fsize):
  
  fsize_2d = (fsize, fsize)
  output_filter = np.ones(fsize_2d)

  output_filter = output_filter/np.sum(output_filter)

  return output_filter


##############################################
#             2D Gaussian Filter1            #
##############################################
def gaussianFilter2D1(fsize, sigma):
  fsize_2d = (fsize, fsize)
  output_filter = np.zeros(fsize_2d)

  fcenter = fsize // 2
  kernal_d = 2 * np.pi * sigma ** 2 # denominator of gaussian

  for h in range(fsize):
    x = h - fcenter
    for w in range(fsize):
      y = w - fcenter
      # numerator of gaussian
      kernal_n = np.exp(-1 * (x ** 2 + y ** 2) / (2 * (sigma ** 2)))
      output_filter[h, w] = kernal_n / kernal_d
      
  output_filter /= np.sum(output_filter) # normalization
  
  return output_filter


##############################################
#             2D Gaussian Filter2            #
##############################################
def gaussianFilter2D2(fsize, sigma):
  fsize_2d = (fsize, fsize)
  output_filter = np.zeros(fsize_2d)

  fcenter = fsize // 2
  # denominator of gaussian
  kernal_d = 2 * np.pi * sigma ** 2 

  # gaussian filter
  gaussian = np.arange(2 * fcenter ** 2 + 1)
  gaussian = np.exp(-1 * gaussian / (2 * (sigma ** 2))) / kernal_d

  for h in range(fsize):
    x = h - fcenter
    for w in range(fsize):
      y = w - fcenter
      output_filter[h, w] = gaussian[x ** 2 + y ** 2]

  output_filter /= np.sum(output_filter) # normalization

  return output_filter



##############################################
#             2D Gaussian Filter3            #
##############################################
def gaussianFilter2D3(fsize, sigma):
  fcenter = fsize // 2
  filter_x = np.array([np.arange(-fcenter, fcenter + 1)])
  filter_y = np.transpose(filter_x)
  output_filter = filter_x ** 2 + filter_y ** 2

  kernal_d = 2 * np.pi * (sigma ** 2)

  gaussian = np.arange(2 * fcenter ** 2 + 1)
  gaussian = np.exp(-1 * gaussian / (2 * (sigma ** 2))) / kernal_d

  output_filter = gaussian[output_filter]
  
  output_filter /= np.sum(output_filter) # normalization
  
  return output_filter



##############################################
#          Separable Gaussian Filter1        #
##############################################
def gaussianFilter1D1(fsize, sigma):
  output_filter = np.zeros((fsize, 1))
  
  fcenter = fsize // 2
  kernal_d = np.sqrt(2 * np.pi) * sigma # denominator of gaussian

  for i in range(fsize):
    x = i - fcenter
    # numerator of gaussian
    kernal_n = np.exp(-1 * (x ** 2) / (2 * (sigma ** 2)))
    output_filter[i] = kernal_n / kernal_d
      
  output_filter /= np.sum(output_filter) # normalization

  return output_filter

##############################################
#          Separable Gaussian Filter2        #
##############################################
def gaussianFilter1D2(fsize, sigma):
  fcenter = fsize // 2
  output_filter = np.array([np.arange(-fcenter, fcenter + 1) ** 2])
  
  # denominator of gaussian
  kernal_d = np.sqrt(2 * np.pi) * sigma

  # compute gaussian
  gaussian = np.arange(fcenter ** 2 + 1)
  gaussian = np.exp(-1 * gaussian / (2 * (sigma ** 2))) / kernal_d
  output_filter = gaussian[output_filter]
  
  # normalization
  output_filter /= np.sum(output_filter)

  return output_filter


##############################################
#           Main Function for HW01-1         #
##############################################

def make(img, fsize, sigma):
  print(f"Image Size = {img.shape}")
  print(f"Filter Size = {fsize} * {fsize}")

  start = time.time()
  gaussian_filter1 = gaussianFilter2D1(fsize, sigma)
  out_gaussian = conv(img, gaussian_filter1)
  end = time.time()
  print(f"RunTime of Filtering with 2D Gaussian = {end - start:.10f}s")

  start = time.time()
  gaussian_filter_v = gaussianFilter1D1(fsize, sigma)
  gaussian_filter_h = np.transpose(gaussian_filter_v)
  out_gaussian = conv(img, gaussian_filter_v)
  out_gaussian = conv(out_gaussian, gaussian_filter_h)
  end = time.time()
  print(f"RunTime of Filtering with 1D Gaussian = {end - start:.10f}s")
  return

img = loadImg('qwer.jpg')
for i in range(10):
  print(f"RunTime {i}")
  make(img, 15, 5)
  print('')
  make(img, 51, 15)
  print('')
  make(img, 101, 35)
  print('')
  make(img, 151, 50)
  print('')
  make(img, 251, 85)
  print('')



'''

start = time.time()
gaussian_filter1 = gaussianFilter2D1(fsize, sigma)
mid = time.time()
print(f"RunTime of Making 2D Gaussian Filter 1 = {mid - start:.10}s")
out_gaussian = conv(img, gaussian_filter1)
end1 = time.time() - start
#print(f"RunTime of Filtering with 2D Gaussian 1 = {end - start:.10f}s")

start = time.time()
gaussian_filter2 = gaussianFilter2D2(fsize, sigma)
mid = time.time()
print(f"RunTime of Making 2D Gaussian Filter 2 = {mid - start:.10}s")
out_gaussian = conv(img, gaussian_filter2)
end2 = time.time() - start
#print(f"RunTime of Filtering with 2D Gaussian 2 = {end - start:.10f}s")

start = time.time()
gaussian_filter3 = gaussianFilter2D3(fsize, sigma)
mid = time.time()
print(f"RunTime of Making 2D Gaussian Filter 3 = {mid - start:.10}s")
out_gaussian = conv(img, gaussian_filter3)
end3 = time.time() - start
#print(f"RunTime of Filtering with 2D Gaussian 3 = {end - start:.10f}s")

print(f"RunTime of Filtering with 2D Gaussian 1 = {end1:.10f}s")
print(f"RunTime of Filtering with 2D Gaussian 2 = {end2:.10f}s")
print(f"RunTime of Filtering with 2D Gaussian 3 = {end3:.10f}s")


'''
'''

##############################################
#           Main Function for HW01-1         #
##############################################

img = loadImg('lenna.png')

fsize = 5 # variable filter size
sigma = 5 # variable variation

for size in range(7, 20, 6):
  fsize = size
  for sig in range(1):
    sigma = fsize // 3
    gaussian_filter = gaussianFilter2D3(fsize, sigma)
    print(f"fsize = {fsize}, sigma = {sigma}")
    print(gaussian_filter)
    out_gaussian = conv(img, gaussian_filter)

    saveImg(out_gaussian, '2d_gauss_lenna_' + str(fsize) + '_' + str(sigma) +'.png')

    
'''
'''

##############################################
#           Main Function for HW01-2         #
##############################################

start = time.time()
gaussian_filter_v = gaussianFilter1D1(fsize, sigma)
gaussian_filter_h = np.transpose(gaussian_filter_v)
mid = time.time()
print(f"RunTime of Making 1D Gaussian Filter 1 = {mid - start:.10}s")

out_gaussian = conv(img, gaussian_filter_v)
out_gaussian = conv(out_gaussian, gaussian_filter_h)
end = time.time()
print(f"RunTime of Gaussian 1D 1 = {end - start:.20f}s")


start = time.time()
gaussian_filter_v = gaussianFilter1D2(fsize, sigma)
gaussian_filter_h = np.transpose(gaussian_filter_v)
mid = time.time()
print(f"RunTime of Making 1D Gaussian Filter 2 = {mid - start:.10}s")

out_gaussian = conv(img, gaussian_filter_v)
out_gaussian = conv(out_gaussian, gaussian_filter_h)
end = time.time()
print(f"RunTime of Gaussian 1D 2 = {end - start:.20f}s")


'''

time.sleep(1)