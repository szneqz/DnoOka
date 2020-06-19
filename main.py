from skimage import data, io, filters, morphology, feature, color, measure, img_as_ubyte
import numpy as np
import wx
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import moment

def calcAccuracy(goodImage, testImage, mask):
    goodAccPix = 0
    goodSensPix = 0
    goodSpecPix = 0
    allPix = goodImage.size
    allBlack = 0
    allWhite = 0
    x = 0
    y = 0
    for a in range(0, allPix - 1):
        if mask[y][x] != 0:
            if goodImage[y][x] == testImage[y][x]:
                goodAccPix += 1
            if goodImage[y][x] == 0:
                allBlack += 1
            else:
                allWhite += 1
            if testImage[y][x] == 0 and goodImage[y][x] == 0:
                goodSpecPix += 1
            if testImage[y][x] == 255 and goodImage[y][x] == 255:
                goodSensPix += 1
        x += 1
        if x >= goodImage.shape[1]:
            x = 0
            y += 1
    output = "Trafnosc: " + str(goodAccPix/allPix) + "\nCzulosc: " + str(goodSensPix/allWhite) + "\nSwoistosc: " \
             + str(goodSpecPix/allBlack) + "\nSrednia arytmetyczna: " + str((goodSensPix/allWhite + goodSpecPix/allBlack) / 2) \
             + "\nSrednia geometryczna: " + str(math.sqrt((goodSensPix/allWhite) * (goodSpecPix/allBlack)))
    return output

def convertOko(inputPath, maskPath):
    mask = io.imread(maskPath, as_gray=True)
    image = io.imread(inputPath)
    image = image[:, :, 1]  # green only
    mask2 = image < 135
    mask = mask * mask2
    image = filters.gaussian(image, sigma=4.5)  # rozmycie o sigmie 4.5
    image = filters.frangi(image, sigmas=0.5)
    image = morphology.dilation(image)  # dylatacja, zwiekszam objetosc
    image = morphology.dilation(image)
    image = image > 0.00000000001
    image = morphology.erosion(image)
    image = morphology.erosion(image)
    image = morphology.dilation(image)
    image = morphology.dilation(image)
    image = filters.gaussian(image, sigma=3.5)  # rozmycie o sigmie 3.5
    image = image > 0.55
    image = image * mask
    image = morphology.remove_small_objects(measure.label(image.copy()), min_size=50000,
                                            connectivity=50000) > 0  # usuwam male obiekty
    image = img_as_ubyte(image)
    io.imsave('output.png', image)


def artificialOko(inputPath, maskPath, inputAIPath, maskAIPath, idealAIPath):
    testImgMask = io.imread(maskPath, as_gray=True)
    testImg = io.imread(inputPath)

    mask = io.imread(maskAIPath, as_gray=True)
    image = io.imread(inputAIPath)
    output = io.imread(idealAIPath, as_gray=True)

    testImg = testImg[:, :, 1]  # green only
    image = image[:, :, 1]  # green only

    # shape zwraca liczbę wierszy i liczbę kolumn
    h, w = testImg.shape[0], testImg.shape[1]

    # rozmiar wycinka = 5 x 5
    PATCH_SIZE = 5
    HALF_PATCH_SIZE = 2

    # io.imshow(image[0])

    LEARN_OFFSET = 2 * PATCH_SIZE
    # Faza uczenia
    answers = []
    features = []

    # j = wiersz
    for j in range(0, h - PATCH_SIZE, LEARN_OFFSET):
        # i = kolumna
        for i in range(0, w - PATCH_SIZE, LEARN_OFFSET):  # szukam punktow wyjsciowych
            if mask[j + HALF_PATCH_SIZE, i + HALF_PATCH_SIZE] == 0:
                continue
            if output[j + HALF_PATCH_SIZE, i + HALF_PATCH_SIZE] > 0:
                answers.append(1)
            else:
                answers.append(0)
            patch = image[j:(j + PATCH_SIZE), i:(i + PATCH_SIZE)]
            m = moment(patch, [2, 3], axis=None)
            features.append([np.mean(patch), m[0], m[1]])

    X = np.array(features)
    X.shape

    y = np.array(answers)

    # mainfeatures = list(zip(features1, features2, features3))
    # KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=10))
    model.fit(X, y)

    TEST_OFFSET = 1
    outputArray = np.zeros((h, w), dtype=np.uint8)

    # j = wiersz
    for j in range(0, h - PATCH_SIZE, TEST_OFFSET):

        X_test = []

        # i = kolumna
        for i in range(0, w - PATCH_SIZE, TEST_OFFSET):  # szukam punktow wyjsciowych
            patch = testImg[j:(j + PATCH_SIZE), i:(i + PATCH_SIZE)]
            m = moment(patch, [2, 3], axis=None)
            X_test.append([np.mean(patch), m[0], m[1]])

        # y_pred = model.predict_proba(np.array(X_test))[:, 1]
        y_pred = model.predict(np.array(X_test))

        for p, i in enumerate(range(0, w - PATCH_SIZE, TEST_OFFSET)):  # szukam punktow wyjsciowych
            if testImgMask[j, i] == 0:
                continue
            outputArray[j + HALF_PATCH_SIZE, i + HALF_PATCH_SIZE] = round(255 * y_pred[p])

    # io.imshow(outputArray)

    io.imsave("output.png", outputArray)

class MainFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MainFrame, self).__init__(*args, **kw)
        self.panel = wx.Panel(self, wx.ID_ANY)

        self.inputPath = 0
        self.maskPath = 0
        self.idealPath = 0
        self.inputAIPath = 0
        self.maskAIPath = 0
        self.idealAIPath = 0

        self.image1toggle = 0
        self.image23toggle = 0

        self.headerText1 = wx.StaticText(self.panel, label="Obraz wejściowy", style=wx.ALIGN_CENTER)
        self.headerText2 = wx.StaticText(self.panel, label="Obraz idealny", style=wx.ALIGN_CENTER)
        self.headerText3 = wx.StaticText(self.panel, label="Obraz wyjściowy", style=wx.ALIGN_CENTER)
        self.pathText1 = wx.StaticText(self.panel,
                                       label="(Plik > Otwórz obraz wejściowy)\n(Plik > Otwórz maskę wejściową)",
                                       style=wx.ALIGN_CENTER)
        self.pathText2 = wx.StaticText(self.panel, label="(Plik > Otwórz idealny obraz wynikowy)", style=wx.ALIGN_CENTER)
        self.pathText3 = wx.StaticText(self.panel, label="(Wybierz metodę przetwarzania)", style=wx.ALIGN_CENTER)
        self.variableText1 = wx.StaticText(self.panel, label="")
        self.variableText2 = wx.StaticText(self.panel, label="")
        self.variableText3 = wx.StaticText(self.panel, label="")
        font = self.headerText1.GetFont()
        font.PointSize += 1
        font = font.Bold()
        self.headerText1.SetFont(font)
        self.headerText2.SetFont(font)
        self.headerText3.SetFont(font)

        fastButton = wx.Button(self.panel, wx.ID_ANY, "Technika przetwarzania obrazu", size=(200, 32))
        fastButton.Bind(wx.EVT_BUTTON, self.onFastButton)
        slowButton = wx.Button(self.panel, wx.ID_ANY, "Technika z użyciem klasyfikatora", size=(200, 32))
        slowButton.Bind(wx.EVT_BUTTON, self.onSlowButton)

        headerSizer = wx.BoxSizer(wx.HORIZONTAL)
        imageSizer = wx.BoxSizer(wx.HORIZONTAL)
        pathSizer = wx.BoxSizer(wx.HORIZONTAL)
        buttonSizer = wx.BoxSizer(wx.HORIZONTAL)
        variableSizer = wx.BoxSizer(wx.HORIZONTAL)
        windowSizer = wx.BoxSizer(wx.VERTICAL)

        self.image1 = wx.Bitmap.FromRGBA(438, 292, alpha=255)
        self.image1mask = wx.Bitmap.FromRGBA(438, 292, alpha=255)
        self.image2 = wx.Bitmap.FromRGBA(438, 292, alpha=255)
        self.image3 = wx.Bitmap.FromRGBA(438, 292, alpha=255)

        self.inputImage = wx.StaticBitmap(self.panel, wx.ID_ANY, bitmap=self.image1)
        self.idealImage = wx.StaticBitmap(self.panel, wx.ID_ANY, bitmap=self.image2)
        self.outputImage = wx.StaticBitmap(self.panel, wx.ID_ANY, bitmap=self.image3)
        self.inputImage.Bind(wx.EVT_LEFT_DOWN, self.onInputClick)
        self.idealImage.Bind(wx.EVT_LEFT_DOWN, self.onOutputClick)
        self.outputImage.Bind(wx.EVT_LEFT_DOWN, self.onOutputClick)

        headerSizer.Add(self.headerText1, 1, wx.ALIGN_CENTER)
        headerSizer.Add(self.headerText2, 1, wx.ALIGN_CENTER)
        headerSizer.Add(self.headerText3, 1, wx.ALIGN_CENTER)
        imageSizer.Add(self.inputImage, 1)
        imageSizer.Add(self.idealImage, 1)
        imageSizer.Add(self.outputImage, 1)
        pathSizer.Add(self.pathText1, 1, wx.ALIGN_CENTER)
        pathSizer.Add(self.pathText2, 1, wx.ALIGN_CENTER)
        pathSizer.Add(self.pathText3, 1, wx.ALIGN_CENTER)
        buttonSizer.Add(fastButton, wx.SizerFlags().Border(wx.ALL, 5))
        buttonSizer.Add(slowButton, wx.SizerFlags().Border(wx.ALL, 5))
        variableSizer.Add(self.variableText1, 1, wx.ALIGN_CENTER)
        variableSizer.Add(self.variableText2, 1, wx.ALIGN_CENTER)
        variableSizer.Add(self.variableText3, 1, wx.ALIGN_CENTER)
        windowSizer.Add(headerSizer, 0, wx.EXPAND | wx.ALL)
        windowSizer.Add(imageSizer, 1, wx.EXPAND | wx.ALL)
        windowSizer.Add(pathSizer, 0, wx.EXPAND | wx.ALL)
        windowSizer.Add(buttonSizer, 1, wx.ALIGN_CENTER)
        windowSizer.Add(variableSizer, 1, wx.EXPAND | wx.ALL)
        # windowSizer.SetSizeHints(self)
        self.panel.SetSizer(windowSizer)

        self.makeMenuBar()
        self.CreateStatusBar()
        self.SetStatusText("")

    def makeMenuBar(self):
        fileMenu = wx.Menu()

        inputImageItem = fileMenu.Append(-1, "&Otwórz obraz wejściowy\tCtrl-O", "Wybierz wejściowy obraz dna oka")
        maskImageItem = fileMenu.Append(-1, "Otwórz &maskę wejściową\tCtrl-M", "Wybierz wejściową maskę obrazu dna oka")
        idealImageItem = fileMenu.Append(-1, "Otwórz &idealny obraz wynikowy\tCtrl-I",
                                         "Wybierz wyjściowy obraz idealny do porównania")
        fileMenu.AppendSeparator()
        inputLearnItem = fileMenu.Append(-1, "Otwórz obraz wejściowy do nauki", "Wybierz wejściowy obraz do nauki SI")
        maskLearnItem = fileMenu.Append(-1, "Otwórz maskę wejściową do nauki", "Wybierz wejściową maskę obrazu do nauki SI")
        idealLearnItem = fileMenu.Append(-1, "Otwórz idealny obraz do nauki",
                                         "Wybierz wyjściowy obraz idealny do nauki SI")
        fileMenu.AppendSeparator()
        outputImageItem = fileMenu.Append(-1, "Zapi&sz obraz wyjściowy\tCtrl-S", "Zapisz wyjściowy obraz naczyń dna oka")
        fileMenu.AppendSeparator()
        exitItem = fileMenu.Append(wx.ID_EXIT)

        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT, "O Progr&amie\tCtrl-A", "Informacje o programie")

        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&Plik")
        menuBar.Append(helpMenu, "P&omoc")

        self.SetMenuBar(menuBar)

        self.Bind(wx.EVT_MENU, self.OnInputOpen, inputImageItem)
        self.Bind(wx.EVT_MENU, self.OnMaskOpen, maskImageItem)
        self.Bind(wx.EVT_MENU, self.OnIdealOpen, idealImageItem)
        self.Bind(wx.EVT_MENU, self.OnInputAIOpen, inputLearnItem)
        self.Bind(wx.EVT_MENU, self.OnMaskAIOpen, maskLearnItem)
        self.Bind(wx.EVT_MENU, self.OnIdealAIOpen, idealLearnItem)
        self.Bind(wx.EVT_MENU, self.OnExit, exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)

    def onOutputClick(self, event):
        if self.image23toggle == 0:
            self.outputImage.Bitmap = self.image2
            self.idealImage.Bitmap = self.image3
            self.headerText3.SetLabel("Obraz idealny")
            self.headerText2.SetLabel("Obraz wyjściowy")
            self.image23toggle = 1
        else:
            self.outputImage.Bitmap = self.image3
            self.idealImage.Bitmap = self.image2
            self.headerText2.SetLabel("Obraz idealny")
            self.headerText3.SetLabel("Obraz wyjściowy")
            self.image23toggle = 0
        self.panel.Layout()

    def onInputClick(self, event):
        if self.image1toggle == 0:
            self.inputImage.Bitmap = self.image1mask
            self.headerText1.SetLabel("Maska obrazu wejściowego")
            self.image1toggle = 1
        else:
            self.inputImage.Bitmap = self.image1
            self.headerText1.SetLabel("Obraz wejściowy")
            self.image1toggle = 0
        self.panel.Layout()

    def onFastButton(self, event):
        errorText = ""
        if self.inputPath == 0:
            errorText += "Otwórz plik wejściowy\n"
        if self.maskPath == 0:
            errorText += "Otwórz maskę wejściową\n"
        if errorText == "":
            convertOko(self.inputPath, self.maskPath)
            self.image3 = wx.Bitmap("output.png").ConvertToImage().Scale(438, 292, wx.IMAGE_QUALITY_HIGH)
            self.image3 = wx.Bitmap(self.image3)
            self.image23toggle = 0
            self.headerText2.SetLabel("Obraz idealny")
            self.headerText3.SetLabel("Obraz wyjściowy")
            self.idealImage.Bitmap = self.image2
            self.outputImage.Bitmap = self.image3

            img2 = io.imread(self.idealPath, as_gray=True)
            img3 = io.imread("output.png", as_gray=True)
            imgmask = io.imread(self.maskPath, as_gray=True)
            dat = calcAccuracy(img2, img3, imgmask)
            self.variableText2.SetLabel(dat)

            self.panel.Layout()
        else:
            wx.MessageBox(errorText)

    def onSlowButton(self, event):
        errorText = ""
        if self.inputPath == 0:
            errorText += "Otwórz plik wejściowy\n"
        if self.maskPath == 0:
            errorText += "Otwórz maskę wejściową\n"
        if self.inputAIPath == 0:
            errorText += "Otwórz obraz wejściowy do nauki\n"
        if self.maskAIPath == 0:
            errorText += "Otwórz maskę wejściową do nauki\n"
        if self.idealAIPath == 0:
            errorText += "Otwórz idealny obraz do nauki\n"
        if errorText == "":
            artificialOko(self.inputPath, self.maskPath, self.inputAIPath, self.maskAIPath, self.idealAIPath)
            self.image3 = wx.Bitmap("output.png").ConvertToImage().Scale(438, 292, wx.IMAGE_QUALITY_HIGH)
            self.image23toggle = 0
            self.headerText2.SetLabel("Obraz idealny")
            self.headerText3.SetLabel("Obraz wyjściowy")
            self.idealImage.Bitmap = self.image2
            self.image3 = wx.Bitmap(self.image3)
            self.outputImage.Bitmap = self.image3

            img2 = io.imread(self.idealPath, as_gray=True)
            img3 = io.imread("output.png", as_gray=True)
            imgmask = io.imread(self.maskPath, as_gray=True)
            dat = calcAccuracy(img2, img3, imgmask)
            self.variableText2.SetLabel(dat)    

            self.panel.Layout()
        else:
            wx.MessageBox(errorText)

    def openFile(self):
        with wx.FileDialog(self, "Wybierz obraz",
                           wildcard="Obrazy (*.png;*.bmp;*.gif;*.jpg;*.jpeg;*.tiff;*.tif)|*.png;*.bmp;*.gif;*.jpg;*.jpeg;*.tiff;*.tif",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return 0
            return fileDialog.GetPath()

    def OnSave(self, event):
        wx.MessageBox("Zapisano")
        print("Zapisano")

    def OnExit(self, event):
        self.Close(True)
        print("Poprawne wyjście z programu")

    def OnInputOpen(self, event):
        self.inputPath = self.openFile()
        if self.inputPath != 0:
            self.image1 = wx.Bitmap(self.inputPath).ConvertToImage().Scale(438, 292, wx.IMAGE_QUALITY_HIGH)
            self.image1 = wx.Bitmap(self.image1)
            self.inputImage.Bitmap = self.image1
            self.headerText1.SetLabel("Obraz wejściowy")
            self.image1toggle = 0
            if self.maskPath == 0:
                self.pathText1.SetLabel(self.inputPath + "\n(Plik > Otwórz maskę wejściową)")
            else:
                self.pathText1.SetLabel(self.inputPath + "\n" + self.maskPath)
            self.panel.Layout()

    def OnMaskOpen(self, event):
        self.maskPath = self.openFile()
        if self.maskPath != 0:
            self.image1mask = wx.Bitmap(self.maskPath).ConvertToImage().Scale(438, 292, wx.IMAGE_QUALITY_HIGH)
            self.image1mask = wx.Bitmap(self.image1mask)
            self.inputImage.Bitmap = self.image1mask
            self.headerText1.SetLabel("Maska obrazu wejściowego")
            self.image1toggle = 1
            if self.inputPath == 0:
                self.pathText1.SetLabel("(Plik > Otwórz obraz wejściowy)\n" + self.maskPath)
            else:
                self.pathText1.SetLabel(self.inputPath + "\n" + self.maskPath)
            self.panel.Layout()

    def OnIdealOpen(self, event):
        self.idealPath = self.openFile()
        if self.idealPath != 0:
            self.image2 = wx.Bitmap(self.idealPath).ConvertToImage().Scale(438, 292, wx.IMAGE_QUALITY_HIGH)
            self.image2 = wx.Bitmap(self.image2)
            self.image23toggle = 0
            self.headerText2.SetLabel("Obraz idealny")
            self.headerText3.SetLabel("Obraz wyjściowy")
            self.outputImage.Bitmap = self.image3
            self.idealImage.Bitmap = self.image2
            self.pathText2.SetLabel(self.idealPath)
            self.panel.Layout()

    def OnInputAIOpen(self, event):
        self.inputAIPath = self.openFile()

    def OnMaskAIOpen(self, event):
        self.maskAIPath = self.openFile()

    def OnIdealAIOpen(self, event):
        self.idealAIPath = self.openFile()

    def OnAbout(self, event):
        wx.MessageBox("Dno Oka v0.1.0\nPUT INF 06/2020\nKrzysztof Schneider\t\t138285\nMieszko Taranczewski\t110225",
                      "O programie",
                      wx.OK | wx.ICON_INFORMATION)


if __name__ == '__main__':
    app = wx.App()
    frame = MainFrame(None, title='Dno oka', size=(1320, 600), style=wx.DEFAULT_FRAME_STYLE ^ (wx.RESIZE_BORDER | wx.MAXIMIZE_BOX))
    frame.Show()
    app.MainLoop()
