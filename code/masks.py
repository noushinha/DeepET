# ============================================================================================
# DeepET - a deep learning framework for segmentation and classification of
#                  macromolecules in Cryo Electron Tomograms (Cryo-ET)
# ============================================================================================
# Copyright (c) 2021 - now
# ZIB - Department of Visual and Data Centric
# Author: Noushin Hajarolasvadi
# Team Leader: Daniel Baum
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================

from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QApplication, QMainWindow
from gui.theme_style import *
from PyQt5.QtGui import QIcon
from gui.mask_generation import mask_generation
from utils.utility_tools import *
from utils.params import *
from gui.mask_generation import OrthosliceWidget

class MaskGenerationwindow(QMainWindow):
    def __init__(self):
        super(MaskGenerationwindow, self).__init__()

        self.ui = mask_generation.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Mask Generation")
        self.setWindowIcon(QIcon('../../icon.jpg'))

        self.dwidget = OrthosliceWidget.OrthoslicesWidget()
        self.ui.gridLayout_5.addWidget(self.dwidget, 0, 0, 1, 1)

        self.extension = 'mrc'
        self.class_radilist = []
        self.class_names = []
        self.class_num = len(self.class_names)
        self.input_image = np.array([])
        self.tomodim = self.input_image.shape
        self.mask_image = []
        self.mask_shape = 'circle'
        self.content = pd.DataFrame()
        self.isNotDragged = False

        self.ui.loadBtn.clicked.connect(self.getfiles)
        self.ui.saveBtn.clicked.connect(self.save_mask)

        self.ui.opacity.valueChanged.connect(self.change_opacity)
        self.set_sldier()
        self.ui.slider.valueChanged.connect(self.change_slide)

        p = self.ui.annTable.palette()
        p.setColor(QPalette.Base, Qt.gray)
        self.ui.annTable.setPalette(p)

    def getfiles(self):
        # download_path = self.ui.inputPath.text()
        # dialog = QFileDialog()
        # dialog.setDirectory(os.path.abspath(__file__))
        selected_file = QFileDialog.getOpenFileName(self, 'Single File', '../data', "CSV files (*.csv);;Text files (*.txt);;XML files (*.xml);;Star files (*.star)")
        self.ui.inputPath.setText(selected_file[0])
        self.readfile(selected_file)

    def getmaskshape(self):
        if self.ui.RecRBtn.isChecked():
            self.mask_shape = 'rectangle'

    def getradiuslength(self):
        radiuslisttext = self.ui.radiLen.text()
        self.class_radilist = radius_list(self.class_names, radiuslisttext)

    def loaddata(self):
        self.ui.annTable.setRowCount(0)
        rows = self.content.shape[0]
        cols = self.content.shape[1]
        for row_number in range(rows):
            self.ui.annTable.insertRow(row_number)
            # self.ui.annTable.setColumnCount(row_number)
            for column_number in range(cols):
                # self.ui.annTable.insertColumn(column_number)
                data = str(self.content[row_number][column_number])
                self.ui.annTable.setItem(row_number, column_number, QTableWidgetItem(str(data)))

    def loadimage(self):
        self.extension = self.content[0][0].split(".")[1]
        self.image_path = os.path.join(ROOT_DIR, self.content[0][0])
        self.readimage()

    def loadmask(self):
        # load the tomogram and coordinates
        self.mask_image = np.zeros(self.tomodim)
        self.isNotDragged = True

        self.getmaskshape()
        self.getradiuslength()

        self.dwidget.lmap = generate_spheres(self.content, self.mask_image, self.class_radilist)

        self.change_slide()
        self.set_opacity()

    def readfile(self, selected_file):
        filepath, filetype = file_attributes(selected_file)

        if filetype == "csv":
            self.content = pd.read_csv(filepath)
        elif filetype == "txt":
            self.content = pd.read_csv(filepath, delimiter="\t")
        elif filetype == "xml":
            self.content = read_xml(filepath)
        elif filetype == "star":
            self.content = read_starfile(filepath)
        self.content = self.content.to_numpy()
        self.loaddata()
        self.loadimage()
        self.getclasses()
        self.loadmask()

    def readimage(self):
        if self.extension == "png":
            self.input_image = cv2.imread(self.image_path)
        elif self.extension == "mrc":
            self.input_image = read_mrc(self.image_path)
        elif self.extension == "tif":
            self.input_image = read_xml(self.image_path)
        elif self.extension == "h5":
            self.input_image = read_starfile(self.image_path)
        else:
            throwErr('ext:'+str(self.extension))
        self.tomodim = self.input_image.shape

    def getclasses(self):
        classes = self.content[:, -1]
        self.class_names = np.unique(classes)
        self.class_num = len(self.class_names)

    def save_mask(self):
        output_path = ROOT_DIR.__str__() + self.ui.outputPath.text()

        # save the generated mask
        filename = 'target_' + self.image_path.split(OS_path_separator)[-1]
        # save_volume(self.dwidget.lmap, output_path + filename + '.png')
        write_mrc(np.array(self.dwidget.lmap).astype(np.int8), output_path + filename)
        print("finished!")

    def set_opacity(self):
        self.dwidget.isLmapLoaded = True
        self.ui.opacity.setMinimum(self.getdatavalue(self.dwidget.vol_mu))
        self.ui.opacity.setMaximum(self.getdatavalue(self.dwidget.vol_max))
        self.ui.opacity.setValue(self.getdatavalue(self.dwidget.levels[1]))

    def getdatavalue(self, val):
        return 100*(val-self.dwidget.vol_min)/(self.dwidget.vol_max-self.dwidget.vol_min)

    def change_opacity(self):
        opacity = float(self.ui.opacity.value()) / 100
        self.dwidget.set_lmap_opacity(opacity)

    def set_sldier(self):
        self.ui.slider.setMaximum(0)
        self.ui.slider.setMaximum(199)  # self.tomodim[0])
        self.ui.slider.setValue(np.round(100))  # self.tomodim[0] / 2
        slide_tootip = "value" + str(self.ui.slider.value())
        self.ui.slider.setToolTip(slide_tootip)
        # self.change_slide()

    def change_slide(self):
        self.dwidget.slide = self.ui.slider.value()
        self.dwidget.set_vol(self.input_image)

        if self.isNotDragged:
            self.dwidget.isColorLoaded = True

        self.dwidget.set_lmap(self.mask_image)
        slide_tootip = "value" + str(self.ui.slider.value())
        self.ui.slider.setToolTip(slide_tootip)
        self.isNotDragged = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_theme_style(app)

    application = MaskGenerationwindow()
    application.show()

    application.showMaximized()

    sys.exit(app.exec_())

# self.graphLayout = pg.GraphicsLayoutWidget()
# self.graphView = self.graphLayout.addViewBox()
# self.image_holder = pg.ImageItem()
# self.graphView.addItem(self.image_holder)
# self.graphView.setRange(QtCore.QRectF(0, 0, 512, 512))
# self.horizontalLayout_11.addWidget(self.graphLayout)