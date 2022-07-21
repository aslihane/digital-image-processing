from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtPrintSupport import QPrintDialog
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUi
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ImageQt import ImageQt
import random
import pygame
import pygame.camera


class windowQT(QMainWindow):
    def __init__(self):
        super(windowQT, self).__init__()
        loadUi('smartvision.ui', self)
        self.open.clicked.connect(self.openCam)
        self.close.clicked.connect(self.closeCam)
        self.snap.clicked.connect(self.takePic)
        self.load.clicked.connect(self.loadImg)
        self.cropsave.clicked.connect(self.cropAndSave)
        self.analysis.clicked.connect(self.imgAnalysis)
        self.help.clicked.connect(self.getHelp)

        self.gray.clicked.connect(self.convertGray)
        self.bw.clicked.connect(self.convertBW)
        self.gaussiannoise.clicked.connect(self.gaussianNoise)
        self.peppersalt.clicked.connect(self.psNoise)
        self.gaussianfilter.clicked.connect(self.gaussianFilter)
        self.mean.clicked.connect(self.meanFilter)
        self.median.clicked.connect(self.medianFilter)
        self.edges.clicked.connect(self.detectEdges)
        self.corners.clicked.connect(self.detectCorners)
        self.fft.clicked.connect(self.ffTransform)

        self.gray.hide()
        self.bw.hide()
        self.mean.hide()
        self.median.hide()
        self.gaussianfilter.hide()
        self.gaussiannoise.hide()
        self.peppersalt.hide()
        self.fft.hide()
        self.edges.hide()
        self.corners.hide()

    def openCam(self):
        self.cap = cv2.VideoCapture(0)

        while True:
            self.ret, self.frame = self.cap.read()
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            Image.fromarray(self.frame).save('c.png')
            self.before.setPixmap(QtGui.QPixmap("c.png"))

            self.c = cv2.waitKey(1)
            # Press ECS to quit
            if self.c == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("Opening the camera")

    def closeCam(self):
        self.clearLabels()
        self.cap = 0

    def takePic(self):
        if self.cap.isOpened():
            Image.fromarray(self.frame).save('snappic.png')
            self.after.setPixmap(QtGui.QPixmap("snappic.png"))

        print("Taking a picture")

    def clearLabels(self):
        self.before.clear()
        self.after.clear()
        self.cap = 0

    def loadImg(self):

        self.clearLabels()

        self.filename = QFileDialog.getOpenFileName(
            self, 'insert image',  'C:\\', 'Image Files (*)')

        self.pngfile = QPixmap(self.filename[0])
        self.before.setPixmap(self.pngfile)
        print("Loading an image")

    def convertGray(self):
        Q_image = QtGui.QPixmap.toImage(self.pngfile)

        grayscale = Q_image.convertToFormat(QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(grayscale)

        self.after.setPixmap(pixmap)

    def convertBW(self):
        Q_image = QtGui.QPixmap.toImage(self.pngfile)
        Q_image.save('temp.png', 'png')
        img = cv2.imread('temp.png')

        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, bwImg) = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)

        Image.fromarray(bwImg).save('temp.png')
        self.after.setPixmap(QtGui.QPixmap("temp.png"))

    def gaussianNoise(self):
        Q_image = QtGui.QPixmap.toImage(self.pngfile)
        Q_image.save('temp.png', 'png')
        img = cv2.imread('temp.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        row, col, ch = img.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        img = img + gauss

        Image.fromarray((img*255).astype(np.uint8)).save('temp.png')
        self.after.setPixmap(QtGui.QPixmap("temp.png"))

    def psNoise(self):
        Q_image = QtGui.QPixmap.toImage(self.pngfile)
        Q_image.save('temp.png', 'png')
        img = cv2.imread('temp.png', cv2.IMREAD_GRAYSCALE)

        row, col = img.shape
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):
            y_coord = random.randint(0, row - 1)
            x_coord = random.randint(0, col - 1)
            img[y_coord][x_coord] = 255

        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):
            y_coord = random.randint(0, row - 1)
            x_coord = random.randint(0, col - 1)
            img[y_coord][x_coord] = 0

        Image.fromarray(img).save('temp.png')
        self.after.setPixmap(QtGui.QPixmap("temp.png"))

    def gaussianFilter(self):
        Q_image = QtGui.QPixmap.toImage(self.pngfile)
        Q_image.save('temp.png', 'png')
        img = cv2.imread('temp.png')

        blur = cv2.GaussianBlur(img, (5, 5), 0)
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

        Image.fromarray(blur).save('temp.png')
        self.after.setPixmap(QtGui.QPixmap("temp.png"))

    def meanFilter(self):
        Q_image = QtGui.QPixmap.toImage(self.pngfile)
        Q_image.save('temp.png', 'png')
        img = cv2.imread('temp.png')

        kernel = np.ones((5, 5), np.float32)/25
        dst = cv2.filter2D(img, -1, kernel)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

        Image.fromarray(dst).save('temp.png')
        self.after.setPixmap(QtGui.QPixmap("temp.png"))

    def medianFilter(self):
        Q_image = QtGui.QPixmap.toImage(self.pngfile)
        Q_image.save('temp.png', 'png')
        img = cv2.imread('temp.png')
        blur = cv2.medianBlur(img, 5)
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

        Image.fromarray(blur).save('temp.png')
        self.after.setPixmap(QtGui.QPixmap("temp.png"))

    def detectEdges(self):
        Q_image = QtGui.QPixmap.toImage(self.pngfile)
        Q_image.save('temp.png', 'png')

        img = cv2.imread('temp.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        edges = cv2.Canny(img, 100, 200)

        Image.fromarray(edges).save('temp.png')
        self.after.setPixmap(QtGui.QPixmap("temp.png"))

    def detectCorners(self):
        Q_image = QtGui.QPixmap.toImage(self.pngfile)
        Q_image.save('temp.jpg')
        image = cv2.imread('temp.jpg')
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        operatedImage = np.float32(operatedImage)

        dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

        dest = cv2.dilate(dest, None)
        img[dest > 0.01 * dest.max()] = [0, 0, 255]

        Image.fromarray(img).save('temp.jpg')
        self.after.setPixmap(QtGui.QPixmap("temp.jpg"))

    def ffTransform(self):
        Q_image = QtGui.QPixmap.toImage(self.pngfile)
        Q_image.save('temp.jpg')
        image = cv2.imread('temp.jpg')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        Image.fromarray(image).save('temp.jpg')
        self.after.setPixmap(QtGui.QPixmap("temp.jpg"))

    def cropAndSave(self):

        print("Cropping and saving the picture")
        Q_image = QtGui.QPixmap.toImage(self.pngfile)
        Q_image.save('temp.jpg')
        im = Image.open('temp.jpg')
        cropped = im.crop((1, 2, 300, 300))
        cropped.save('cropped.jpg')

        self.after.setPixmap(QtGui.QPixmap("cropped.jpg"))

    def imgAnalysis(self):

        print("Analysing the image")
        self.gray.show()
        self.bw.show()
        self.mean.show()
        self.median.show()
        self.gaussianfilter.show()
        self.gaussiannoise.show()
        self.peppersalt.show()
        self.fft.show()
        self.edges.show()
        self.corners.show()

        if self.cap.isOpened():
            self.closeCam()
            self.before.setPixmap(QtGui.QPixmap("snappic.png"))

    def getHelp(self):
        print("Helping")
        QMessageBox.about(self, "Help", "CONNECT: You can connect your camera, turn it off or take snapshots. \n IMAGE PROCESSING: You can browse images, crop them and apply filters. \n If you need more help, contact us via smartvision@xxxx.com")


if __name__ == "__main__":
    app = QApplication([])
    window = windowQT()
    window.setWindowTitle('Smart Vision')
    window.show()
    sys.exit(app.exec_())
    app.exec_()
    exit.clicked.connect(app.exit)
    exit.show()
