import cv2 as cv 
import numpy as np


class CreateRectange():
    def __init__(self, imagePath, windowName, imread_method=-1, save=False):
        self.imagePath = imagePath 
        self.windowName = windowName 
        self.x_array = []
        self.y_array = []
        self.count = 0
        self.imread_method = imread_method
        self.save = save

    def run(self):
        while True:
            # Load Image
            self.image = cv.imread(self.imagePath, self.imread_method)

            # Showing images
            cv.imshow(self.windowName, self.image)

            cv.setMouseCallback(self.windowName, self.onclick)

            if cv.waitKey(0) == 27:
                cv.destroyAllWindows()
                break

    def onclick(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print('Left Button: {}, {}'.format(x, y))
            cv.setMouseCallback(self.windowName, self.mousemove)

    def mousemove(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE:
            print('Mouse Moving: {}, {}'.format(x, y))

            # Draw a point
            cv.circle(self.image, (x,y), 5, (255, 255, 255), -1)
            
            # Show the image 
            cv.imshow(self.windowName, self.image)

            # Adding each mousemove point to a array and getting the max and min x,y 
            self.xmax, self.ymax, self.xmin, self.ymin = self.getMaxMinXY(x, y)


        if event == cv.EVENT_LBUTTONDOWN:
            print('Left Button Moving: {}, {}'.format(x, y))

            # Drawing the rectangle with max and min value
            cv.rectangle(self.image, (self.xmax, self.ymax), (self.xmin, self.ymin), (0, 255, 255), 2)
            cv.imshow(self.windowName, self.image)
            
            # Clearning the array for the job is done
            self.x_array.clear()
            self.y_array.clear()
            print('Cleared Array!')

            if self.save:
                croped_image = self.image[self.ymin+2:self.ymax-2, self.xmin+2:self.xmax-2]
                croped_image = cv.cvtColor(croped_image, cv.COLOR_BGR2GRAY)

                # Saving the cropped image
                cv.imwrite('image/cropped{}.jpg'.format(self.count), croped_image)
                print('Image Saved!')
                self.count += 1
            cv.setMouseCallback(self.windowName, self.onclick)

    def getMaxMinXY(self, x, y):
        self.x_array.append(x)
        self.y_array.append(y)
        
        # Getting xmax and ymax
        xmax = np.max(self.x_array) + 20
        ymax = np.max(self.y_array) + 20

        # Getting xmin and ymin 
        xmin = np.min(self.x_array) - 20
        ymin = np.min(self.y_array) - 20

        return (xmax, ymax, xmin, ymin)

            

if __name__ == "__main__":
    c = CreateRectange('image/white.jpg', 'Draw', save= True)
    c.run()