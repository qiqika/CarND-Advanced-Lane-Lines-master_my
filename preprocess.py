import numpy as np
import cv2
import matplotlib.pyplot as plt
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Create a copy and apply the threshold
    gradbinary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    gradbinary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
   
    # Return the result
    return gradbinary_output
   

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    magbinary_output = np.zeros_like(gradmag)
    magbinary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return magbinary_output

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dirbinary_output =  np.zeros_like(absgraddir)
    dirbinary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return dirbinary_output

def HLStogray(image):
    gray0 = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    gray = gray0[:,:,2] 
    return gray
def YUVtogray(image):
    gray0 = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    gray = gray0[:,:,0] 
    return gray

def color_threshold(image, thresh=(0, 255)):
    gray = HLStogray(image)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
    #fig  = plt.figure(); ax = fig.add_subplot(1,1,1)
    #ax.imshow(binary, cmap='gray')
    
    gray1 = YUVtogray(image)
    binary1 = np.zeros_like(gray1)
    binary1[(gray1 > thresh[0]) & (gray1 <= thresh[1])] = 1
    
    #fig1  = plt.figure(); ax1 = fig1.add_subplot(1,1,1)
    #ax1.imshow(binary1, cmap='gray')
    
    combined = np.zeros_like(gray1)
    combined[ (binary1 == 1)&(binary ==1) ] = 1#
    return combined

def select_white_yellow(image):
 
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
 
    # white color mask
    lower = np.uint8([  0, 200,   0])
 
    upper = np.uint8([255, 255, 255])
 
    white_mask = cv2.inRange(converted, lower, upper)
 
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
 
    upper = np.uint8([ 40, 255, 255])
 
    yellow_mask = cv2.inRange(converted, lower, upper)
 
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
 
    return cv2.bitwise_and(image, image, mask = mask)

    
def extract_image(image, ksize):
    #kernel_size =9
    #image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    image = select_white_yellow(image)
#-----------------------color space-----------------------
    binaryx = color_threshold(image, thresh=(0, 255))
    #fig0  = plt.figure(); ax0 = fig0.add_subplot(1,1,1)
    #ax0.imshow(binaryx, cmap='gray')
    #binaryx = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # -----------------------------sobel operator dx dy----------------------
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(10, 150))
    #fig1  = plt.figure(); ax1 = fig1.add_subplot(1,1,1)
    #ax1.imshow(gradx, cmap='gray')

    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(10, 150))
    #fig2  = plt.figure(); ax2 = fig2.add_subplot(1,1,1)
    #ax2.imshow(grady, cmap='gray')
    #--------------------------------sobel operator distance dx**2+dy**2 ------------------
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(10, 150))
    #fig3  = plt.figure(); ax3 = fig3.add_subplot(1,1,1)
    #ax3.imshow(mag_binary , cmap='gray')
    #--------------------------------sobel angle-------------------------------
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.9, 1.0))
    #fig4  = plt.figure(); ax4 = fig4.add_subplot(1,1,1)
    #ax4.imshow(dir_binary , cmap='gray')

    combined = np.zeros_like(dir_binary)
    combined[(binaryx ==1 )|(((gradx == 1) & (grady == 1)) | ((mag_binary == 1)& (dir_binary == 1)))] = 1
    #fig5  = plt.figure(); ax5 = fig5.add_subplot(1,1,1)(binaryx ==1 )|
    #ax5.imshow(combined, cmap='gray')

   #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))  
    #opend = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    opend= combined
    fig6  = plt.figure(); ax6 = fig6.add_subplot(1,1,1)
    ax6.imshow(opend, cmap='gray')
    return opend

def perspective_image(image):
    img_size = (image.shape[1],image.shape[0])
    imshape = image.shape
    #print(img_size)
    # For source points I'm grabbing the outer four detected corners
    point1 = [int(190/960*imshape[1]),imshape[0]]
    point2 = [int(450/960*imshape[1]), int(340/540*imshape[0])]
    point3 = [int(530/960*imshape[1]),int(340/540*imshape[0])]
    point4 = [int(900/960*imshape[1]),imshape[0]]
    src = np.float32([point1, point2, point3, point4 ])
    #print(src)
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    
    point11 = [190/960*imshape[1],imshape[0]]
    point12 = [190/960*imshape[1],0]
    point13 = [900/960*imshape[1],0]
    point14 = [900/960*imshape[1],imshape[0]]
    dst = np.float32([point11, point12, point13, point14 ])
   
    #print(src)
    # Given src and dst points, calculate the perspective transform matrix
    Minv = cv2.getPerspectiveTransform(dst,src )
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, M, img_size)


    #fig8  = plt.figure(); ax8 = fig8.add_subplot(1,1,1)
    #ax8.imshow(warped, cmap='gray')
    return warped,Minv