# In[1]:


#get_ipython().magic('matplotlib inline')
import cv2
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
x=1
# In[2]:
  
def smaller_img(img):
    #resizes the video
    #scale_percent=0.5
    width=int(img.shape[1]/4)
    height=int(img.shape[0]/4)
    dimension=(width,height)
    return cv2.resize(img,dimension,interpolation=cv2.INTER_AREA);

def find_biggest_contour(image):
    # check OpenCV version
    major = cv2.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Copy to prevent modification
    #image = image.copy()
    #img, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #print len(contours)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
 
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask


# In[3]:
    
cap = cv2.VideoCapture('tomato_video.mp4')

while(cap.isOpened()):
    ret, oframe = cap.read()
    
#cap = cv2.VideoCapture(0)
#
#while(1):
#
#    # Take each frame
#    _, oframe = cap.read()
    
    frame=smaller_img(oframe)
    
    # Convert BGR to LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

# In[4]:
   
    # define range of red color in LAB
    lower_red = np.array([0,153,0])
    upper_red = np.array([255,255,255])
    
    # Threshold the LAB image to get only red colors
    image_red = cv2.inRange(lab, lower_red, upper_red)

    
# In[5]:
    
    ## Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    #
    ## Fill small gaps
    image_red_closed = cv2.morphologyEx(image_red, cv2.MORPH_CLOSE, kernel)
    #simage_red_closed=smaller_img(image_red_closed) 
    #cv2.imshow('image_red_closed',image_red_closed)
    
    ## Remove specks
    image_red_closed_then_opened = cv2.morphologyEx(image_red_closed, cv2.MORPH_OPEN, kernel)
    #image_red_closed_then_opened=smaller_img(image_red_closed_then_opened) 
    cv2.imshow('image_red_closed_then_opened',image_red_closed_then_opened)
    
    
# In[6]:

    big_contour, red_mask = find_biggest_contour(image_red_closed_then_opened)
    cv2.imshow('biggest_contour',red_mask)
    
# In[58]:

    # Bounding ellipse
    image_with_ellipse = frame.copy()
    ellipse = cv2.fitEllipse(big_contour)
    cv2.ellipse(image_with_ellipse, ellipse, (0,255,0), 2)
    cv2.imshow('image_with_ellipse',image_with_ellipse)
    
    
    # In[59]:

    #  Bounding Box
    ret,thresh = cv2.threshold(red_mask,-1,255,-1)
    
    # check OpenCV version
    major = cv2.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    image_boundbox=frame.copy()
    
    for item in range(len(contours)):
        cnt = contours[item]
        if len(cnt)>20:
            #print(len(cnt))
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(image_boundbox,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow('image_boundbox',image_boundbox)
    
        
# In[60]:

    # Centre of mass
    mom = cv2.moments(red_mask)
    
    # Calculating x,y coordinate of center
    if mom['m00'] != 0:
        cX = int(mom['m10']/mom['m00'])
        cY = int(mom['m01']/mom['m00'])
    else:
        cX,cY = 0, 0
    
    image_with_com = frame.copy()
    cv2.circle(image_with_com, (cX , cY), 5, (0, 255, 0), -1)
    cv2.imshow('image_with_com',image_with_com)

# In[7]:   
 
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= image_red)
    
    #sframe=smaller_img(frame)
    cv2.imshow('frame',frame)
    #cv2.imshow('mask',image_red)
    #sres=smaller_img(res)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()