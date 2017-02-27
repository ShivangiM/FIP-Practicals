
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import cv2
import numpy as np 
from matplotlib import pyplot as plt

# in opencv imread converts image to nparray therefore the shape works


# In[2]:

import os
cwd = os.getcwd()
print cwd


# In[3]:

print os.listdir(cwd)


# In[4]:

img = cv2.imread('bonsai.jpg')
plt.imshow(img)


# In[5]:

resized_image = cv2.resize(img,(100,100)) 
plt.imshow(resized_image)


# In[6]:

print "Size of original image: {}".format(img.shape)
print "Size of resized image:  {}".format(resized_image.shape)


# In[7]:

img1 = cv2.imread('daisy.jpg')
im_pl = plt.imshow(img1)


# In[8]:

img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#plt.imshow(img_gray)
im_pl_gr = plt.imshow(img_gray,cmap='Greys')


# In[9]:

print img_gray.shape


# In[10]:

from PIL import Image


# In[11]:

image_file = Image.open("daisy.jpg")
plt.imshow(image_file)# open colour image


# In[12]:

image_file_gray = image_file.convert('1') # convert image to black and white
plt.imshow(image_file_gray)


# In[13]:

print image_file.size


# In[14]:

img_gray.shape


# In[15]:

img_gray_profile =img_gray[200,:]


# In[16]:

img_gray_profile.shape


# In[17]:

#plt.show(img_gray_profile)
len(img_gray_profile)


# In[18]:

#IMAGE PROFILE
s = np.arange(0,len(img_gray_profile),1)
plt.plot(s, img_gray_profile)


# In[19]:

img_chan = img1[:,:,1]


# In[20]:

plt.imshow(img_chan)


# In[21]:

b,g,r = cv2.split(img1)


# In[22]:

plt.imshow(b, cmap="gray")


# In[23]:

plt.imshow(g, cmap="gray")


# In[24]:

plt.imshow(r, cmap='gray')


# In[25]:

#back_img1 = 0.2989 * r + 0.5870 * g + 0.1140 * b 
#back_img1.shape


# In[26]:

img1_back = cv2.merge((b,g,r))
plt.imshow(img1_back)
print "Size of merged image: {}".format(img1_back.shape)


# In[27]:

#print img1_back


# In[28]:

neg_img1 = 255 - img1
plt.imshow(neg_img1)


# In[29]:

#flip_img1 = cv2.flip(img)
#plt.imshow(flip_img1,0)


# In[30]:

img=cv2.imread('daisy.jpg')
rimg=img.copy()
fimg=img.copy()
rimg = cv2.flip(img,1)
fimg = cv2.flip(img,0)
cv2.imwrite('flip-vertical.jpg',fimg)
cv2.imwrite('horizontal-flip.jpg', rimg)


# In[31]:

#img_flipv = cv2.imread('flip-vertical.jpg')
#plt.imshow(img_flipv)
#img_fliph = cv2.imread('horizontal-flip.jpg')
#plt.imshow(img_fliph)
plt.imshow(img)


# In[32]:

#LOOP
image = cv2.imread('cat.jpg',0)
plt.imshow(image)

for i in xrange(image.shape[0]):
    for j in xrange(image.shape[1]):
        image[i] = 100 - image[i]
        
plt.imshow(image, cmap='gray')


# In[33]:

#Thersholding
img = cv2.imread('cat.jpg',0)
img = cv2.medianBlur(img,5)
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,            cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


# In[34]:

#Contrast Streching
img = cv2.imread('daisy.jpg',0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

'''You might recall that the cumulative distribution function is defined for discrete random variables as:

F(x)=P(X≤x)=∑t≤xf(t)'''


# 3 and 4


# In[35]:

img = cv2.imread('daisy.jpg',0)
equ = cv2.equalizeHist(img)


# In[36]:

res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('res.jpg',res)


# In[37]:

im = cv2.imread('res.jpg')
plt.imshow(im)


# In[38]:

image_daisy = cv2.imread('daisy.jpg',0)
image_daisy = cv2.resize(image_daisy, (300, 235)) 
image_daisy.shape


# In[39]:

image_cat = cv2.imread('cat.jpg',0)
image_cat.shape


# In[40]:

add_images = image_daisy + image_cat
plt.imshow(add_images,cmap='gray')

for i in xrange(add_images.shape[0]):
    for j in xrange(add_images.shape[1]):
        if add_images[i][j]>255:
            add_images[i][j]=255


# In[41]:

plt.imshow(add_images,cmap='gray')


# In[42]:

sub_images = image_daisy - image_cat
plt.imshow(sub_images,cmap='gray')


# In[43]:

for i in xrange(sub_images.shape[0]):
    for j in xrange(sub_images.shape[1]):
        if sub_images[i][j]<0:
            sub_images[i][j]=0
plt.imshow(add_images,cmap='gray')


# In[44]:

image_daisy = cv2.imread('daisy.jpg',0)
image_daisy = cv2.resize(image_daisy, (300, 235))
image_cat = cv2.imread('cat.jpg',0)
average_image = np.zeros((len(image_daisy[0]), len(image_daisy[1])))
for i in xrange(image_daisy.shape[0]):
    for j in xrange(image_daisy.shape[1]):
        average_image[i][j] = (2*image_daisy[i][j]+ image_cat[i][j])/2
plt.imshow(average_image, cmap='gray')


# In[45]:

image_cat_not = cv2.bitwise_not(image_cat)
plt.imshow(image_cat_not,cmap='gray')


# In[47]:

for i in xrange(image_daisy.shape[0]):
    for j in xrange(image_daisy.shape[1]):
        new_image_and = image_daisy & image_cat
#img1_bg = cv2.bitwise_and(image_cat,image_cat)
plt.imshow(new_image_and, cmap='gray')


# In[48]:

for i in xrange(image_daisy.shape[0]):
    for j in xrange(image_daisy.shape[1]):
        new_image_or = image_daisy | image_cat
#img_or = cv2.bitwise_or(image_daisy,image_cat)
plt.imshow(new_image_or,cmap='gray')


# In[49]:

for i in xrange(image_daisy.shape[0]):
    for j in xrange(image_daisy.shape[1]):
        new_image_xor = image_daisy^image_cat
#img_xor = cv2.bitwise_xor(image_daisy,image_cat)
plt.imshow(new_image_xor,cmap='gray')


# In[ ]:



