import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

import math

from tkinter import *  # create buttons
from tkinter import filedialog

# import PIL.Image
from PIL import ImageTk, ImageDraw, Image




def resize_image(
        img):  # Urmatorul pas a fost redimensionarea imaginii. Acesta este necesar din doua motive, dupa caz. Fie pentru a reduce numarul de pixeli din imagine si pentru a scadea timpul de executie al programului, astfel avand o complexitate mai redusa, fie pentru a mari o imagine.
    print('Original Dimensions : ',
          img.shape)  # OpenCV ofera mai multe metode de interpolare pentru redimensionarea unei imagini, iar cele pe care am ales sa le utilizez au fost cv2.INTER_AREA pentru a micsora o imagine, cv2.INTER_LINEAR pentru a mari o imagine.
    # tbd: fie marirea imaginii, fie micsorarea
    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim,
                         interpolation=cv2.INTER_AREA)  # Este de retinut faptul ca atunci cand este folosita functia cv2.resize, tuplul pasat pentru a determina dimensiunea noii imagini (#tbd în acest caz) urmează ordinea (lățimea, înălțimea), spre deosebire de cele așteptate (înălțime, lățime).
    # https://www.geeksforgeeks.org/image-resizing-using-opencv-python/

    print('Resized Dimensions : ', resized.shape)
    return resized


image = cv2.imread("hair.jpg")
image = resize_image(image)

#https://stackoverflow.com/questions/20169137/opencv-with-python-specific-filter-not-same
def kirsch_filter(gray):
    if gray.ndim > 2:
        raise Exception("illegal argument: input must be a single channel image (gray)")
    kernelG1 = np.array([[5, 5, 5],
                         [-3, 0, -3],
                         [-3, -3, -3]], dtype=np.float32)
    kernelG2 = np.array([[5, 5, -3],
                         [5, 0, -3],
                         [-3, -3, -3]], dtype=np.float32)
    kernelG3 = np.array([[5, -3, -3],
                         [5, 0, -3],
                         [5, -3, -3]], dtype=np.float32)
    kernelG4 = np.array([[-3, -3, -3],
                         [5, 0, -3],
                         [5, 5, -3]], dtype=np.float32)
    kernelG5 = np.array([[-3, -3, -3],
                         [-3, 0, -3],
                         [5, 5, 5]], dtype=np.float32)
    kernelG6 = np.array([[-3, -3, -3],
                         [-3, 0, 5],
                         [-3, 5, 5]], dtype=np.float32)
    kernelG7 = np.array([[-3, -3, 5],
                         [-3, 0, 5],
                         [-3, -3, 5]], dtype=np.float32)
    kernelG8 = np.array([[-3, 5, 5],
                         [-3, 0, 5],
                         [-3, -3, -3]], dtype=np.float32)

    g1 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g2 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g3 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG3), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g4 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG4), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g5 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG5), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g6 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG6), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # g7 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG7), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # g8 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    magn = cv2.max(
        g1, cv2.max(
            g2, cv2.max(g3, cv2.max(g4, g5)
                        )
        )
    )

    return magn


def kirsch_call(image):
    fg = image
    fg_rgb = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2GRAY)

    bin = kirsch_filter(gray)
    return bin


# site: https://pythonspot.com/tk-window-and-button/
# https://docs.python.org/3/library/tk.html


root = Tk()
root.minsize(300, 100)
root.geometry("320x100")


def myClick():
    print("click!")
    global folder_path
    filename = filedialog.askopenfilename()
    folder_path.set(filename)
    print(filename)


def tkinter_buttons():
    folder_path = StringVar()
    lbl1 = Label(master=root, textvariable=folder_path)
    lbl1.grid(row=0, column=1)
    print(folder_path)
    myButton = Button(root, text="Add Photo", command=myClick())
    myButton.grid(row=0, column=3)

    head, tail = os.path.split(str(folder_path))
    print(tail)
    print(head)

    im = Image.open(tail).convert("RGB")
    im.resize(300, 300)
    tkimage = ImageTk.PhotoImage(im)
    imageButton = Label(root, image=tkimage)
    image.grid(row=0, column=5)

    root.mainloop()



def skin():
    img = cv2.imread('man.jpg')
    new_image = img
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            pixel = new_image[i, j]
            skin = pixel[2] > 95 and pixel[1] > 40 and pixel[0] > 20 and \
                   (int(max(pixel) - min(pixel)) > 15) and abs(int(pixel[2] - pixel[1])) > 15 and \
                   pixel[2] > pixel[1] and pixel[2] > pixel[0]
            if not skin:
                new_image[i][j] = [0, 0, 0]
    cv2.imshow('method_1', new_image)
    cv2.imwrite("method_1.jpg", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def coordonatele_se_afla_in_interiorul_dreptunghiului(image, i,
                                                      j):  # https://stackoverflow.com/questions/54400034/what-does-cv2s-detectmultiscale-method-return - crops the initial image into faces' images seperately
    faceCascade = cv2.CascadeClassifier(
        'C:\\Users\\ocu02060\\Documents\\New folder\\Licenta_Samson_Georgiana\\licenta2\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier(
        'C:\\Users\\ocu02060\\Documents\\New folder\\Licenta_Samson_Georgiana\\licenta2\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("faces", faces)

    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if i >= x and i <= x + w and j >= y and j <= y + h:
            return 1
        else:
            return 0


# detect_face_region(image, k=0)


def discharge_pixels_from_face_region():
    #
    print("eliminare pixeli de pe fata")


def find_pixel_color(contours, img):
    #
    lst_intensities = []

    # For each list of contour points...
    for i in range(len(contours)):
        # Create a mask image that contains the contour filled in
        cimg = np.zeros_like(img)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

        # Access the image pixels and create a 1D numpy array then add to list
        pts = np.where(cimg == 255)
        lst_intensities.append(img[pts[0], pts[1]])

    f = open("list_intensities", "w")
    f.write(str(lst_intensities))
    print("gaseste culoarea unui pixel si coloreaza alti pixeli cu aceeasi culoare")


def find_biggest_contour(image):
    list_contours = []


    # color boundaries [B, G, R]
    lower = [1, 0, 10]
    upper = [60, 60, 200]

    # create NumPy arrays from boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    #mask = kirsch_call(image)
    cv2.imshow("Mask", mask)

    cv2.waitKey(0)

    output = cv2.bitwise_and(image, image, mask=mask)

    #cv2.imshow("Output", output)

    #cv2.waitKey(0)

    # threshold value
    #ret, thresh = cv2.threshold(mask, 50,255, cv2.THRESH_BINARY)

    #am folosit treshold adaptativ
    thresh = cv2.adaptiveThreshold(mask, 40, cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY, 11, 3)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(output, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # try to find by perimeter but doesn't work
        # perimeter = max(contours, key = cv2.arcLength(contours, False))

        # x,y,w,h = cv2.boundingRect(c)

        # draw the biggest contour (c) in magenta
        img_copy = image.copy()

        new_image = img_copy
        final = cv2.drawContours(new_image, contours, contourIdx=-1, color=(255, 0, 255), thickness=3)
        print(contours)
        contours = np.array(contours).tolist()

        for sublist in contours:
            for item in sublist:
                list_contours.append(item[0])

        poly = np.zeros((512, 512, 3), dtype="uint8")
        list_contours = np.array(list_contours)

        img_mod = cv2.polylines(poly, [list_contours], True, (255, 120, 255), 3)
        cv2.imshow("polygon", img_mod)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # https://stackoverflow.com/questions/13574751/overlay-polygon-on-top-of-image-in-python
        """list_contours= list(zip(*list_contours))

        x_contours = list_contours[0]
        y_contours = list_contours[1]

        x_contours = map(int, x_contours)
        y_contours = map(int, y_contours)

        img2 = img_copy.copy()
        draw = ImageDraw.Draw(Image.fromarray(img2))
        draw.polygon(zip(x_contours, y_contours), fill="wheat")

        img3 = Image.blend(img_copy, img2, 0.5)
        img3.save('output.png')
"""

        """for i in range(new_image.shape[0]):
            for j in range(new_image.shape[1]):
                print(new_image.shape)
                pixel = new_image[i, j]
                skin = pixel[2] > 95 and pixel[1] > 40 and pixel[0] > 20 and \
                       (int(max(pixel) - min(pixel)) > 15) and abs(int(pixel[2] - pixel[1])) > 15 and \
                       pixel[2] > pixel[1] and pixel[2] > pixel[0]
                if skin and coordonatele_se_afla_in_interiorul_dreptunghiului(new_image, i , j):
                    print(i, " ", j)
                    new_image[i,j] = [0, 0, 0]
        """

        f = open("out", "w")
        f.write(str(contours))  # https://www.geeksforgeeks.org/python-unzip-a-list-of-tuples/  tbd

        # cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

        return image, new_image, output, contours


image, img_copy, output, contours = find_biggest_contour(image)
#find_pixel_color(contours, image)

# show the images
cv2.imshow("Result", np.hstack([image, output]))#Stack arrays in sequence horizontally (column wise).

cv2.waitKey(0)
cv2.imshow("Result", np.hstack([img_copy, output]))
cv2.waitKey(0)
