from skimage import data, io, filters, morphology, feature, color, measure, img_as_ubyte
import numpy as np
import wx
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import moment

globalModel = 0

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


def artificialOko():

    mainPath = "D:/STUDIA/6 semestr/InformatykaWMedycynie/Projekt2/"

    mask = []
    image = []
    output = []

    for i in range(1, 7):
        if i < 10:
            mask.append(io.imread(mainPath + "mask/0" + str(i) + "_h_mask.tif", as_gray=True))
            image.append(io.imread(mainPath + "images/0" + str(i) + "_h.jpg"))
            output.append(io.imread(mainPath + "manual1/0" + str(i) + "_h.tif", as_gray=True))
        else:
            mask.append(io.imread(mainPath + "mask/" + str(i) + "_h_mask.tif", as_gray=True))
            image.append(io.imread(mainPath + "images/" + str(i) + "_h.jpg"))
            output.append(io.imread(mainPath + "manual1/" + str(i) + "_h.tif", as_gray=True))

    for i in range(1, 7):
        image[i - 1] = image[i - 1][:, :, 1]  # green only

    # shape zwraca liczbę wierszy i liczbę kolumn
    h, w = image[0].shape[0], image[0].shape[1]

    # rozmiar wycinka = 5 x 5
    PATCH_SIZE = 5
    HALF_PATCH_SIZE = 2

    # io.imshow(image[0])

    LEARN_OFFSET = 2 * PATCH_SIZE
    # Faza uczenia
    answers = []
    features = []
    nr = -1

    # j = wiersz
    for j in range(0, h - PATCH_SIZE, LEARN_OFFSET):
        # i = kolumna
        for i in range(0, w - PATCH_SIZE, LEARN_OFFSET):  # szukam punktow wyjsciowych
            nr = nr + 1
            if nr > 5:
                nr = 0
            if mask[nr][j + HALF_PATCH_SIZE, i + HALF_PATCH_SIZE] == 0:
                continue
            if output[nr][j + HALF_PATCH_SIZE, i + HALF_PATCH_SIZE] > 0:
                answers.append(1)
            else:
                answers.append(0)
            patch = image[nr][j:(j + PATCH_SIZE), i:(i + PATCH_SIZE)]
            m = moment(patch, [2, 3], axis=None)
            features.append([np.mean(patch), m[0], m[1]])

    X = np.array(features)
    X.shape

    y = np.array(answers)

    # mainfeatures = list(zip(features1, features2, features3))
    # KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

    param_grid = {
        'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }

    rfc = RandomForestClassifier(n_estimators=10)

    GSCV = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    global globalModel
    globalModel = make_pipeline(StandardScaler(), GSCV)

    globalModel.fit(X, y)

    print(GSCV.best_estimator_)


def Predictor(inputPath, maskPath):
    testImgMask = io.imread(maskPath, as_gray=True)
    testImg = io.imread(inputPath)

    testImg = testImg[:, :, 1]  # green only

    # shape zwraca liczbę wierszy i liczbę kolumn
    h, w = testImg.shape[0], testImg.shape[1]

    # rozmiar wycinka = 5 x 5
    PATCH_SIZE = 5
    HALF_PATCH_SIZE = 2

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
        y_pred = globalModel.predict(np.array(X_test))

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
        learnButton = wx.Button(self.panel, wx.ID_ANY, "Uczenie klasyfikatora", size=(200, 32))
        learnButton.Bind(wx.EVT_BUTTON, self.onLearnButton)
        self.slowButton = wx.Button(self.panel, wx.ID_ANY, "Technika z użyciem klasyfikatora", size=(200, 32))
        self.slowButton.Bind(wx.EVT_BUTTON, self.onSlowButton)
        self.slowButton.Disable()

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
        buttonSizer.Add(learnButton, wx.SizerFlags().Border(wx.ALL, 5))
        buttonSizer.Add(self.slowButton, wx.SizerFlags().Border(wx.ALL, 5))
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

    def onLearnButton(self, event):
        artificialOko()
        self.slowButton.Enable()

    def onSlowButton(self, event):
        errorText = ""
        if self.inputPath == 0:
            errorText += "Otwórz plik wejściowy\n"
        if self.maskPath == 0:
            errorText += "Otwórz maskę wejściową\n"
        if errorText == "":
            Predictor(self.inputPath, self.maskPath)
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

    def OnAbout(self, event):
        wx.MessageBox("Dno Oka v0.1.0\nPUT INF 06/2020\nKrzysztof Schneider\t\t138285\nMieszko Taranczewski\t110225",
                      "O programie",
                      wx.OK | wx.ICON_INFORMATION)


if __name__ == '__main__':
    app = wx.App()
    frame = MainFrame(None, title='Dno oka', size=(1320, 600), style=wx.DEFAULT_FRAME_STYLE ^ (wx.RESIZE_BORDER | wx.MAXIMIZE_BOX))
    frame.Show()
    app.MainLoop()
