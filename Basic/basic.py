import cv2 
import numpy as np 
import datetime
import matplotlib.pyplot as plt

# Tutorial 1
"""img = cv2.imread('sample/lena.jpg', 1)
cv2.imshow('image', img)
esc = cv2.waitKey(0) 

if esc == 27:
    cv2.destroyAllWindows()
elif esc == ord('s'):
    cv2.imwrite('sample/lena_copy.png', img)
    cv2.destroyAllWindows()
"""


# Tutorial 2 (Live stream from camera)
"""

# 0 -1 2 3 for the webcame or devices selection
cap = cv2.VideoCapture(-1)

# Saving the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Frame/outputgrey.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()

    # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        out.write(frame)
        cv2.imshow("frame", gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else: 
        break

cap.release()
out.release()
cv2.destroyAllWindows()"""


# Tutorial 3 (Different shape in cv2)
"""# img = cv2.imread('sample/lena.jpg', 1)

# Create image using numpy zeros method
img = np.zeros([512, 512, 3], np.uint8)

img = cv2.line(img, (0, 0), (500, 255), (255, 120, 0), 50) # Line
img = cv2.arrowedLine(img, (0, 0), (500, 255), (255, 120, 0), 50) # Arrowed line
img = cv2.rectangle(img, (0,0), (255, 255), (255, 255, 0), 5) # Rectangle 
img = cv2.circle(img, (255,255), 60, (255, 0,0), 2) # Circle 

font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, 'OpenCV', (100,100), font, 5, (255, 32, 40), 10, cv2.LINE_AA)
cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()"""


# Tutotial 4 (different property)
"""cap = cv2.VideoCapture(0)

print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


# Setting the property as every property has it's number associated
cap.set(3, 1080) # For width
cap.set(4, 720) # For height

# print(cap.get(3))  
# print(cap.get(4))

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'Width: {} Height: {}'.format(cap.get(3), cap.get(4))
        datet = str(datetime.datetime.now())
        frame = cv2.putText(frame, datet, (10, 20), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break 

cap.release()
cv2.destroyAllWindows()"""

# Tutorial 5 (Mouse Event)
"""
events = [i for i in dir(cv2) if 'EVNET' in i]

output = ['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 
         'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 
         'EVENT_FLAG_RBUTTON', 'EVENT_FLAG_SHIFTKEY', 
         'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 
         'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK', 
         'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 
         'EVENT_MOUSEHWHEEL', 'EVENT_MOUSEMOVE', 
         'EVENT_MOUSEWHEEL', 'EVENT_RBUTTONDBLCLK', 
         'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']

def click_event(event, x, y, flags, param):
    
    if event == cv2.EVENT_RBUTTONDOWN:
        print(x,',', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ',' + str(y)
        cv2.putText(img, strXY, (x, y), font, 0.5, (0, 255, 255), 1 )
        cv2.line(img, (x,y), (x,y), (0, 255, 0), 2)
        cv2.imshow('image', img)

    if event == cv2.EVENT_MBUTTONDOWN:
         blue = img[y, x, 0]
         green = img[y, x, 1]
         red = img[y, x, 2]
         font = cv2.FONT_HERSHEY_SIMPLEX
         strBGR = str(blue) + ',' + str(green) + ',' + str(red)
         cv2.putText(img, strBGR, (x,y), font, .5, (0, 0, 255), 2)
         cv2.imshow('Channel', img)

    if event == cv2.EVENT_LBUTTONDOWN:
        # cv2.circle(img, (x,y), 3, (255, 155, 155), -1)
        # points.append((x,y))
        # if len(points) >=2:
        #     cv2.line(img, points[-1], points[-2], (255, 255, 255), 5)
        # cv2.imshow('image', img)
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        cv2.circle(img, (x,y), 3, (0,0,0), -1)
        mycolorImage = np.zeros((512,512,3), dtype=np.uint8)
        print(mycolorImage)
        mycolorImage[:] = [blue, green, red]
        print(mycolorImage)
        cv2.imshow('color', mycolorImage)

# img = np.zeros((512, 512, 3), dtype=np.uint8)
img = cv2.imread('sample/lena.jpg', -1)
cv2.imshow('image', img)
points = []

cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

# Tutorial 6 (basic function ROI: region of intrest)

"""img = cv2.imread('sample/lena.jpg', -1)
img2 = cv2.imread('sample/img2.jpg', -1)

print(img.shape)
print(img.size)
print(img.dtype)

b, g, r = cv2.split(img)
# print('Img BGR: \nB:{}\nG:{}\nR:{}'.format(b, g, r))
img = cv2.merge((b, g, r))

ball = img[280:340, 330:390]
img[273:333, 100:160] = ball
print('Ball:{}'.format(ball))
# print(img)

# Resizing the img
img = cv2.resize(img, (512, 512))
img2 = cv2.resize(img2, (512, 512))

img3 = cv2.add(img, img2)
img4 = cv2.addWeighted(img, .9, img2, .1, 1)
cv2.imshow('image', img3)
cv2.imshow('image2', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()"""



# Default callback function for setMouseCallback()

"""def callback(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		blue = npData[x, y , 0]
		green = npData[x, y, 1]
		red = npData[x, y, 2]

		newColor = np.zeros((512, 512, 3), dtype=np.uint8)

		newColor[:] = [blue, green , red]

		cv2.imshow('data3', newColor)

npData = np.random.rand(512, 512, 3)
npData1 = np.random.randn(512, 512, 3)

cv2.setMouseCallback('sata', callback)
cv2.imshow('data', npData)
cv2.imshow('data1', npData1)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

# Tutorial 7 (Bitwise, AND, OR, NOT and XOR)
"""
img1 = np.zeros((250, 500, 3), dtype=np.uint8)
img1 = cv2.rectangle(img1, (200, 0), (300, 100), (255, 255, 255), -1)
img2 = cv2.imread('sample/image1.jpg', -1)

def create_black_white_image():
    image1 = np.zeros((250, 500, 3), dtype=np.uint8)
    image1 = cv2.rectangle(image1, (250,0), (500, 250), (255,255, 255), -1)
    cv2.imwrite('sample/image1.jpg', image1)

# bitAnd = cv2.bitwise_and(img2, img1)
# bitOr = cv2.bitwise_or(img2, img1)
# bitXor = cv2.bitwise_xor(img2, img1)
# bitNot1 = cv2.bitwise_not(img1)
# bitNot2 = cv2.bitwise_not(img2)

# cv2.imshow('bitAnd', bitAnd)
# cv2.imshow('bitOr', bitOr)
# cv2.imshow('bitXor', bitXor)
# cv2.imshow('bitNot1', bitNot1)
# cv2.imshow('botNot2', bitNot2)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

# Tutorial 8 (Trackbar in opencv)
"""# img = np.zeros((300, 512, 3), np.uint8)

cv2.namedWindow('image')

# Create trackbar
def nothing(x):
    print(x)

cv2.createTrackbar('CP', 'image', 10, 400, nothing)
# cv2.createTrackbar('G', 'image', 0, 255, nothing)
# cv2.createTrackbar('R', 'image', 0, 255, nothing)


# Switch using Trackbar
switch = 'Color/Gray'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

while(1):
    img = cv2.imread('sample/lena.jpg', -1)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break 
    # Get the trackbar value
    pos = cv2.getTrackbarPos('CP', 'image')
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(pos), (50, 50), font, 1, (0, 0, 255), 2)
    
    # g = cv2.getTrackbarPos('G', 'image')
    # r = cv2.getTrackbarPos('R', 'image')
    s = cv2.getTrackbarPos(switch, 'image')
    
    if s == 0:
        # img[:] = 0
        # continue 
        pass  
    else: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img[:] = [b, g ,r]
    cv2.imshow('image', img)
cv2.destroyAllWindows()"""

# Tutorial 9 (Object detection and Object Tracking Using HSV Color Space)

"""def nothing(x):
    pass 

cv2.namedWindow('Tracking')

while True:
    frame = cv2.imread('sample/s.jpeg', -1)
    frame = cv2.resize(frame, (512, 512))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_b = np.array([110, 50, 50])
    u_b = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break 

cv2.destroyAllWindows()
"""
# Tutorial 10 (Thresholding)
"""
image = cv2.imread('sample/gradient.jpeg', -1)
image = cv2.resize(image, (500, 500))

_, th1 = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY_INV)
_, th3 = cv2.threshold(image, 50, 255, cv2.THRESH_TRUNC)
_, th4 = cv2.threshold(image, 20, 255, cv2.THRESH_TOZERO)
_, th5 = cv2.threshold(image, 20, 255, cv2.THRESH_TOZERO_INV)


cv2.imshow('image', image)
cv2.imshow('th1', th1)
cv2.imshow('th2', th2)
cv2.imshow('th3', th3)
cv2.imshow('th4', th4)
cv2.imshow('th5', th5)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# Tutorial 11 (Adaptive Thresold)

# image = cv2.imread('sample/lena.jpg', -1)

# _, th1 = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)

# cv2.imshow('imge', image)
# cv2.imshow('th1',th1)
# cv2.imshow('th2', th2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Tutorial 13 (Using Matplotlib in cv )
"""
image = cv2.imread('sample/gradient.jpeg', -1)
image = cv2.resize(image, (500, 500))

_, th1 = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY_INV)
_, th3 = cv2.threshold(image, 50, 255, cv2.THRESH_TRUNC)
_, th4 = cv2.threshold(image, 20, 255, cv2.THRESH_TOZERO)
_, th5 = cv2.threshold(image, 20, 255, cv2.THRESH_TOZERO_INV)

title = ['Original', 'binary', 'binary-invert', 'trunc', 'tozero', 'tozero-invert']
image = [image, th1, th2, th3, th4, th5]


for i in range(6):
    print(i)
    plt.subplot(2, 3, i+1)
    plt.imshow(image[i], 'gray')
    plt.title(title[i])
    plt.xticks([])
    plt.yticks([])

plt.show()"""

# Tutorial 14 (Dilation)

"""image = cv2.imread('sample/s.jpeg', 0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

_, mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

kernal = np.ones((3,3), dtype=np.uint8)

dilation = cv2.dilate(mask, kernal, iterations=3)
erosion = cv2.erode(mask, kernal, iterations= 2)

# Opening erison folled by dilation
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

# Closing dilation followed by erosion
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)

gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernal)
title = ['original', 'mask', 'dilation' , 'erosion', 'opening', 'closing' , 'gradient']
images = [image, mask, dilation, erosion, opening, closing, gradient]

for i in range(len(title)):
    plt.subplot(3, 3, i+1)
    plt.title(title[i])
    plt.imshow(images[i])
    plt.xticks([])
    plt.yticks([])

plt.show()"""



# Tutorial 15 (Smothing and bluring image)
"""
HINT = 'Low pass filter helps to blur or smooth the image where as high pass filter used to detect the edges'


image = cv2.imread('sample/lena.jpg', -1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (300, 50))

kernal = np.ones((5,5), np.float32)/25
dst = cv2.filter2D(image, -1, kernal)
blur = cv2.blur(image, (5, 5))
gausblur = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_WRAP)
median_salt_paper = cv2.medianBlur(image, 5)

# Preserve the edges
bilateralFilter = cv2.bilateralFilter(image, 9, 75, 775)

title = ['image', '2D Convolutions', 'blur', 'gauseblur', 'Salt_paper', 'bilateralFilter']
images = [image, dst, blur, gausblur, median_salt_paper, bilateralFilter]


for i in range(len(title)):
    plt.subplot(2, 3, i+1)
    plt.title(title[i])
    plt.imshow(images[i])
    plt.xticks([])
    plt.yticks([])


plt.show()"""


# Tutorial 16 ( Image Gradient and Edg Detection)

image = cv2.imread('sample/lena.jpg', -1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply Laplacian
lap = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
# print(lap)
lap = np.uint8(np.absolute(lap))

# Applying the filter
kernal = np.ones((5,5), np.uint8)/25
median_salt_paper = cv2.medianBlur(lap, 3)

# Apply sobelx and sobely
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Canny edge
canny = cv2.Canny(image, 100, 200)

sobelx = np.uint8(np.absolute(sobelx))
sobely = np.uint8(np.absolute(sobely))

# Combile sobelx and sobely
sobelCombine = cv2.bitwise_or(sobelx, sobely)

title = ['Original', 'Laplacian', 'SobelX', 'SobelY', 'SobelCombinCannye', 'Canny', 'median_salt_paper']
images = [image, lap, sobelx, sobely, sobelCombine, canny, median_salt_paper]

for i in range(len(title)):
    plt.subplot(3, 3, i+1)
    plt.title(title[i])
    plt.imshow(images[i])
    plt.xticks([])
    plt.yticks([])

plt.show()


