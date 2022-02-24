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
import numpy as np
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
        self.modified_tomo = np.array([])
        self.tomodim = self.input_image.shape
        self.mask_image = []
        self.mask_shape = 'circle'
        self.content = pd.DataFrame()
        self.isNotDragged = False

        self.ui.loadBtn.clicked.connect(self.getfiles)
        self.ui.saveBtn.clicked.connect(self.save_mask)

        self.ui.opacity.valueChanged.connect(self.change_opacity)
        self.set_slider(True)
        self.ui.slider.valueChanged.connect(self.change_slide)

        p = self.ui.annTable.palette()
        p.setColor(QPalette.Base, Qt.gray)
        self.ui.annTable.setPalette(p)

    def getfiles(self):
        # download_path = self.ui.inputPath.text()
        # dialog = QFileDialog()
        # dialog.setDirectory(os.path.abspath(__file__))
        selected_file = QFileDialog.getOpenFileName(self, 'Single File', '../data',
                                                    "CSV files (*.csv);;Star files (*.star);;"
                                                    "Text files (*.txt);;XML files (*.xml)")
        self.ui.inputPath.setText(selected_file[0])
        self.readfile(selected_file)

    def getmaskshape(self):
        if self.ui.RecRBtn.isChecked():
            self.mask_shape = 'rectangle'

    def getradiuslength(self):
        radiuslisttext = self.ui.radiLen.text()
        self.class_radilist = radius_list(self.class_names, radiuslisttext)
        print(self.class_radilist)

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
        # generate a tomogram for masks to be written on
        self.mask_image = np.zeros(self.tomodim, dtype=np.uint16)
        output_path = ROOT_DIR.__str__() + self.ui.outputPath.text()
        # filename = 'target_' + self.image_path.split(OS_path_separator)[-1]
        # write_mrc(self.mask_image, output_path + filename)
        self.isNotDragged = True

        self.getmaskshape()
        self.getradiuslength()

        # self.mask_image, self.modified_tomo = generate_spheres(self.content, self.mask_image, self.input_image,
        # self.class_radilist)
        self.mask_image = generate_spheres(self.content, self.mask_image, self.input_image, self.class_radilist)
        self.dwidget.lmap = self.mask_image
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
        else:
            throwErr('ext:'+str(self.extension))
        self.tomodim = self.input_image.shape

    def getclasses(self):
        classes = self.content[:, -1]
        self.class_names = np.unique(classes)
        self.class_num = len(self.class_names)

    def save_mask(self):
        # import mrcfile
        output_path = ROOT_DIR.__str__() + self.ui.outputPath.text()

        # save the generated mask
        filename = 'target_' + self.image_path.split(OS_path_separator)[-1]
        # filename2 = 'modified_' + self.image_path.split(OS_path_separator)[-1]
        # write_mrc(np.array(self.dwidget.lmap).astype(np.uint16), output_path + filename)
        write_mrc(self.mask_image, output_path + filename)
        # self.plot_vol(np.array(self.dwidget.lmap).astype(np.int16), output_path)
        # write_mrc(self.modified_tomo, output_path + filename2)
        # save_volume(self.dwidget.lmap, output_path + filename + '.png')

        # tomo_patch = self.input_image[248:448, 1000:1512, 1000:1512]
        # mytensor = np.array(tomo_patch, dtype=np.int8)
        # write_mrc(mytensor, output_path + 'tomo_patch_8.mrc')
        # mrctensor = mrcfile.new_mmap(output_path + filename, shape=self.tomodim, mrc_mode=0)
        # for val in range(len(mrctensor.data)):
        #     mrctensor.data[val] = val
        # write_mrc(mytensor, output_path + filename)

        print("finished!")

    def plot_vol(self, vol_array, output_path):
        """
            save a file from slices of a volume array.
            If volume is int8, the function plots labelmap in color scale.
            otherwise the function consider the volume as a tomogram and plots in gray scale.

            inputs: vol_array: a 3D numpy array
                    filename: '/path/to/output png file'
        """

        # Get central slices along each dimension:
        zindx = int(np.round(vol_array.shape[0] / 2))
        yindx = int(np.round(vol_array.shape[1] / 2))
        xindx = int(np.round(vol_array.shape[2] / 2))

        xy_slice = vol_array[zindx, :, :]  # the xy plane
        zx_slice = vol_array[:, yindx, :]  # the zx plane
        zy_slice = vol_array[:, :, xindx]  # the zy plane

        if vol_array.dtype == np.int8:
            fig1 = plt.figure(num=1, figsize=(10, 10))
            plt.imshow(xy_slice, cmap='jet', vmin=np.min(vol_array), vmax=np.max(vol_array))
            fig2 = plt.figure(num=2, figsize=(10, 5))
            plt.imshow(zx_slice, cmap='jet', vmin=np.min(vol_array), vmax=np.max(vol_array))
            fig3 = plt.figure(num=3, figsize=(5, 10))
            plt.imshow(np.flipud(np.rot90(zy_slice)), cmap='jet', vmin=np.min(vol_array), vmax=np.max(vol_array))
        else:
            mu = np.mean(vol_array)  # mean of the volume/tomogram
            std = np.std(vol_array)  # standard deviation of the volume/tomogram
            fig1 = plt.figure(num=1, figsize=(10, 10))
            plt.imshow(xy_slice, cmap='gray', vmin=mu - 5 * std, vmax=mu + 5 * std)
            fig2 = plt.figure(num=2, figsize=(10, 5))
            plt.imshow(zy_slice, cmap='gray', vmin=mu - 5 * std, vmax=mu + 5 * std)
            fig3 = plt.figure(num=3, figsize=(5, 10))
            plt.imshow(zx_slice, cmap='gray', vmin=mu - 5 * std, vmax=mu + 5 * std)

        # fig1.savefig(os.pat
        #
        #
        #
        #
        # h.join(output_path, "labelmap_xy_plane.png"))
        # fig2.savefig(os.path.join(output_path, "labelmap_zx_plane.png"))
        # fig3.savefig(os.path.join(output_path, "labelmap_zy_plane.png"))
        plt.show()

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

    def set_slider(self, flag):
        self.ui.slider.setMinimum(0)
        if flag:
            self.ui.slider.setMaximum(199)
            self.ui.slider.setValue(100)
        else:
            self.ui.slider.setMaximum(int(self.tomodim[0]))
            self.ui.slider.setValue(int(self.tomodim[0]/2))
        slide_tootip = "value" + str(self.ui.slider.value())
        self.ui.slider.setToolTip(slide_tootip)
        # self.change_slide()

    def change_slide(self):
        self.set_slider(False)
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