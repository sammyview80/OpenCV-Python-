import cv2 as cv 
import numpy as np 


# Creating the windows name 
windowName = 'Drawing'
cv.namedWindow(windowName)

# Creating the white 512, 512 white image using np
image = np.zeros((512, 512, 3), dtype=np.uint8)
cv.imwrite('../Images/white.jpg', image)

# Some Global Variables
areas = []
whs = []
hw = {}
count = 0


# Callback function 
def onclick(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.setMouseCallback(windowName, mouseMove)
        
def areasHeightsWidths(areas, whs, x, y):
    areas.append(x)
    whs.append(y)
    xMax = np.max(areas)
    yMax = np.max(whs)

    xMin = np.min(areas)
    yMin = np.min(whs)
    print('max:')
    print(xMax, yMax)
    print('min:')
    print(xMin, yMin)

    return (xMax, yMax), (xMin, yMin)


def mouseMove(event, x, y, flags, param):
    global areas 
    global whs, hw, count
    
    if event == cv.EVENT_MOUSEMOVE:
        # Drawing a circle
        cv.circle(image, (x,y), 3, (255, 255, 255), -1)
        cv.imshow(windowName, image)

        (xMax, yMax), (xMin, yMin) = areasHeightsWidths(areas, whs, x, y)
        hw['xMax'] = xMax + 20 # Adding padding
        hw['yMax'] = yMax + 20
        hw['xMin'] = xMin - 20
        hw['yMin'] = yMin - 20

    if event == cv.EVENT_LBUTTONDOWN:

        cv.rectangle(image, (hw['xMax'], hw['yMax']), (hw['xMin'], hw['yMin']),(0, 255, 255), 3)
        cv.imshow(windowName, image)
        areas.clear()
        whs.clear()
        x = hw['xMin']
        y = hw['yMin']
        w = hw['xMax'] - hw['xMin']
        h = hw['yMax'] - hw['yMin']

        # Getting the rectangle image
        croped_image = image[ y+3:y+h-3 , x+3:x+w -3] # -3 and +3 for removing rectangle

        # Converting into gray
        croped_image = cv.cvtColor(croped_image, cv.COLOR_BGR2GRAY)

        # Saving into folder
        cv.imwrite('image/Croped{}.jpg'.format(count), croped_image)
        count += 1

        cv.setMouseCallback(windowName, onclick)
def run():
    while True:
        # Load the image
        image = cv.imread('image/white.jpg', -1)

        # Show the image 
        cv.imshow(windowName, image)

        cv.setMouseCallback(windowName, onclick)

        if cv.waitKey(0) == 27:
            cv.destroyAllWindows()
            break

if __name__ == "__main__":
    run()