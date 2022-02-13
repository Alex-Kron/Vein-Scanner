import codecs
import math
import threading
import tkinter as tk

import cv2
import serial
import numpy as np
import mediapipe as mp
from skimage.morphology import skeletonize
from PIL import Image, ImageTk

CAP_WIDTH = 640
CAP_HEIGHT = 480

port = serial.Serial('/dev/ttyUSB0', 9600)
web = cv2.VideoCapture(-1)
web.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
web.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
web.set(cv2.CAP_PROP_FPS, 30)
web.set(cv2.CAP_PROP_GAMMA, 100)
web.set(cv2.CAP_PROP_BRIGHTNESS, 120)

mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils


def stream(label, grayscale, v_pattern):
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as hands:
        while web.isOpened():
            ret, img = web.read()
            if ret:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.flip(img, axis=1)
                img_hand = img
                if grayscale.get():
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if v_pattern.get():
                    results = hands.process(img_hand)
                    if results.multi_hand_landmarks:
                        idx = 0
                        for hand_landmarks in results.multi_hand_landmarks:
                            for idx, landmark in enumerate(hand_landmarks.landmark):
                                if idx == 5:
                                    x1 = landmark
                                if idx == 17:
                                    x2 = landmark
                                if idx == 0:
                                    x3 = landmark
                                if idx == 1:
                                    x4 = landmark
                            x1 = to_ing_coordinates(x1.x, x1.y, CAP_WIDTH, CAP_HEIGHT)
                            x2 = to_ing_coordinates(x2.x, x2.y, CAP_WIDTH, CAP_HEIGHT)
                            x3 = to_ing_coordinates(x3.x, x3.y, CAP_WIDTH, CAP_HEIGHT)
                            x4 = to_ing_coordinates(x4.x, x4.y, CAP_WIDTH, CAP_HEIGHT)
                            mask = np.ones(img_hand.shape, dtype=np.uint8)
                            mask.fill(255)
                            roi_corners = np.array([x2, x1, x4, x3], dtype=np.int32)
                            cv2.fillPoly(mask, np.int32([roi_corners]), 0)
                            mask_img = cv2.bitwise_or(img_hand, mask)
                            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
                            mask_img[mask_img == 255] = 0
                            cv2.imwrite('mask.png', mask_img)
                            bin = get_pattern(mask_img, 150, 250, 29, 1)
                            for b_size_idx in range(29, 39, 2):
                                 tmp_bin = get_pattern(mask_img, 100, 200, b_size_idx, 1)
                                 mat = np.ones((3, 3), 'uint8')
                                 tmp_bin = cv2.dilate(tmp_bin, mat)
                                 bin = cv2.bitwise_and(bin, tmp_bin)
                            bin_skel = bin
                            #bin_skel = cv2.dilate(bin_skel, mat)
                            #contours, _ = cv2.findContours(bin_skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            #img = cv2.drawContours(np.array(img), contours, -1, (255, 0, 0), 2)
                            bin_skel[bin_skel == 255] = 1
                            bin_skel = skeletonize(bin_skel)
                            bin_skel = bin_skel.astype('uint8') * 255
                            img = cv2.add(cv2.cvtColor(img_hand, cv2.COLOR_BGR2GRAY), bin_skel)
            frame_img = ImageTk.PhotoImage(Image.fromarray(img))
            label.config(image=frame_img)
            label.image = frame_img


def create_gabor_filter(ksize, sigma, lambd, gamma):
    filters = []
    num_filters = 16
    # ksize = 21
    # sigma = 3.0
    # lambd = 9.0
    # gamma = 0.2
    psi = 0
    for theta in np.arange(0, np.pi, np.pi / num_filters):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()
        filters.append(kern)
    return filters


def get_pattern(img, th1, th2, b_size, const):
    gray = img #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (17, 17), cv2.BORDER_DEFAULT)
    canny = cv2.Canny(blurred, th1, th2)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, b_size, const)
    # canny = cv2.Canny(thresh, 150, 150)
    return thresh


def apply_filter(img, filters):
    newimage = img
    depth = -1
    for kern in filters:
        # image_filter = cv2.filter2D(img, depth, kern)

        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        # np.maximum(newimage, image_filter, newimage)
        newimage = cv2.erode(newimage, kern)
    return newimage


def to_ing_coordinates(x, y, height, width):
    x = min(math.floor(x * width), width - 1)
    y = min(math.floor(y * height), height - 1)
    return x, y


def palm_range(label):
    while port.isOpen():
        sensor = port.readline()
        sensor = codecs.decode(sensor, 'UTF-8').split("\n")
        label["text"] = f"Distance to palm: {sensor[0]} см"


def vein_pattern(gray):
    pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Palm Scanner")

        # Menu vars
        self.grayscale = tk.BooleanVar()
        self.v_pattern = tk.BooleanVar()
        self.points = tk.BooleanVar()
        self.rectangle = tk.BooleanVar()

        # Menu
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="Save pattern as png...")
        menu.add_cascade(label="File", menu=file_menu)

        capture_menu = tk.Menu(menu, tearoff=0)
        capture_menu.add_checkbutton(label="Grayscale", onvalue=1, offvalue=0, variable=self.grayscale)
        capture_menu.add_checkbutton(label="Vein Pattern", onvalue=1, offvalue=0, variable=self.v_pattern)
        menu.add_cascade(label="Capture", menu=capture_menu)

        roi_menu = tk.Menu(menu, tearoff=0)
        roi_menu.add_checkbutton(label="Anchor points", onvalue=1, offvalue=0, variable=self.points)
        roi_menu.add_checkbutton(label="Rectangle", onvalue=1, offvalue=0, variable=self.rectangle)
        menu.add_cascade(label="Region of interest", menu=roi_menu)

        self.config(menu=menu)

        # Frames
        self.main_frame = tk.Frame(
            master=self,
            relief=tk.FLAT,
            width=1200,
            height=720
        )
        self.main_frame.pack()

    def capture(self):
        cap_frame = tk.Frame(
            master=self.main_frame,
            relief=tk.FLAT
        )
        cap_label = tk.Label(master=cap_frame)
        cap_label.pack()
        cap_thread = threading.Thread(target=stream, args=(cap_label, self.grayscale, self.v_pattern))
        cap_thread.daemon = 1
        cap_frame.pack(side=tk.LEFT)
        cap_thread.start()

    def __del__(self):
        web.release()


if __name__ == "__main__":
    # img = cv2.imread("g_test.jpg")
    # k_size = range(1, 41, 8)
    # sigma = np.arange(1.0, 5.0, 1.0)
    # lambd = range(1, 16, 3)
    # gamma = np.arange(0.1, 1.5, 0.3)
    # for k in k_size:
    #     for s in sigma:
    #         for l in lambd:
    #             for g in gamma:
    #                 filters = create_gabor_filter(k, s, l, g)
    #                 gabour = apply_filter(img, filters)
    #                 cv2.imwrite('res_%d_%f_%d_%f.png' % (k, s, l, g), gabour)
    # for b_size in range(3, 39, 2):
    #    for const in range(0, 5):
    # result = get_pattern(img, 100, 200, 3, 0)
    # for i in range(3, 35, 2):
    #     tmp_result = get_pattern(img, 100, 200, i, 0)
    #     result = cv2.bitwise_and(result, tmp_result)
    #     result_skel = result
    #     result_skel = cv2.dilate(result_skel, np.ones((5, 5), 'uint8'))
    #     result_skel[result_skel == 255] = 1
    #     result_skel = skeletonize(result_skel)
    #     result_skel = result_skel.astype('uint8') * 255
    #     result_img = cv2.add(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), result_skel)
    #     cv2.imwrite('pattern5_%d.png' % i, result_img)

    app = App()
    app.capture()
    app.mainloop()

# window = tk.Tk()
# window.title("Palm Scanner")

# Menu Bar
# grayscale = tk.BooleanVar()
# v_pattern = tk.BooleanVar()
# points = tk.BooleanVar()
# rectangle = tk.BooleanVar()
#
# menuBar = tk.Menu(window)
# fileMenu = tk.Menu(menuBar, tearoff=0)
# fileMenu.add_command(label="Save pattern as png...")
# fileMenu.add_cascade(label="File", menu=menuBar)
#
#
# window.config(menu=menuBar)
###############################################################################################
# frame1 = tk.Frame(
#     master=window,
#     relief=tk.FLAT,
# )
# frame1.grid(row=0, column=0, padx=5, pady=5)
# vsLabel = tk.Label(master=frame1, text=f"Video stream")
# vsLabel.pack()
#
# frame2 = tk.Frame(
#     master=window,
#     relief=tk.FLAT,
# )
# frame2.grid(row=1, column=0, padx=5, pady=5)
# videoLabel = tk.Label(master=frame2)
# videoLabel.pack()
# thread1 = threading.Thread(target=stream, args=(videoLabel,))
# thread1.daemon = 2
# thread1.start()
#
# frame3 = tk.Frame(
#     master=window,
#     relief=tk.FLAT,
# )
# frame3.grid(row=0, column=1, padx=5, pady=5)
# rsLabel = tk.Label(master=frame3, text=f"ROI stream")
# rsLabel.pack()
#
# frame4 = tk.Frame(
#     master=window,
#     relief=tk.FLAT,
# )
# frame4.grid(row=2, column=0, padx=5, pady=5)
# distLabel = tk.Label(master=frame4)
# distLabel.pack()
# thread4 = threading.Thread(target=palm_range, args=(distLabel,))
# thread4.daemon = 1
# thread4.start()
#
# window.mainloop()
# web.release()
