# ============================================================================================
# DeepTomo - a deep learning framework for segmentation and classification of
#                  macromolecules in Cryo Electron Tomograms (Cryo-ET)
# ============================================================================================
# Copyright (c) 2021 - now
# ZIB - Department of Visual and Data Centric
# Author: Noushin Hajarolasvadi, Willy (Daniel team)
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================

from PyQt5 import QtWidgets
from gui.theme_style import *
from PyQt5.QtGui import QIcon
from gui.mask_generation import mask_generation
from utils.utility_tools import *
from utils.params import *
from gui.mask_generation import OrthosliceWidget

class MaskGenerationwindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MaskGenerationwindow, self).__init__()

        self.ui = mask_generation.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Mask Generation")
        self.setWindowIcon(QIcon('../../icon.jpg'))

        self.dwidget = OrthosliceWidget.DisplayOrthoslicesWidget()
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

        self.ui.loadBtn.clicked.connect(self.getfiles)
        self.ui.saveBtn.clicked.connect(self.save_mask)

        p = self.ui.annTable.palette()
        p.setColor(QPalette.Base, Qt.gray)
        self.ui.annTable.setPalette(p)

    def getfiles(self):
        selected_file = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', '../data', "CSV files (*.csv);;Text files (*.txt);;XML files (*.xml);;Star files (*.star)")
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
                self.ui.annTable.setItem(row_number, column_number, QtWidgets.QTableWidgetItem(str(data)))

    def loadimage(self):
        self.extension = self.content[0][0].split(".")[1]
        self.image_path = os.path.join(ROOT_DIR, self.content[0][0])
        self.readimage()

    def loadmask(self):
        # load the tomogram and coordinates
        self.dwidget.set_vol(self.input_image)
        self.mask_image = np.zeros(self.tomodim)
        self.dwidget.set_lmap(self.mask_image)

        self.getmaskshape()
        self.getradiuslength()

        # generate spheres and map them to the 2D image
        self.dwidget.lmap = generate_with_spheres(self.content, self.mask_image, self.class_radilist)
        self.dwidget.update_lmap(self.dwidget.lmap)
        # coord = [100, 256, 256]
        # self.dwidget.goto_coord(coord)


        # self.dwidget.goto_coord(coord)
        # xy_plane, zx_plane, zy_plane = get_planes(self.input_image)
        # xy_plane = normalize_image(xy_plane)
        # xy_plane = g2c_image(xy_plane)
        # for row_number in range(rows):
        #     coord1 = self.content[row_number][3]
        #     coord2 = self.content[row_number][2]
        #
        #     classcolor = tuple(boxcolor[self.content[row_number, -1]][0])
        #     classcolor = (int(classcolor[0]), int(classcolor[1]), int(classcolor[2]))
        #     # radilen = int(self.class_radilist[self.content[row_number][-1]])

            # if self.mask_shape == "circle":
                # target_image = cv2.circle(target_image, (coord1, coord2), radilen, color=classcolor, thickness=-1)
                # cv2.circle(g2c_image(self.dwidget.img_xy.image), (coord1, coord2), radilen, color=classcolor, thickness=-1)
                # self.dwidget.img_xy.image[][]
            # elif self.mask_shape == "rectangle":
            #     top_left_x, top_left_y = coord2 - radilen // 2, coord1 - radilen // 2
            #     bottom_right_x, bottom_right_y = coord2 + radilen // 2, coord1 + radilen // 2
                # target_image = cv2.rectangle(target_image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y),
                #                              color=classcolor, thickness=-1)
                # xy_plane = cv2.rectangle(target_image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y),
                #                              color=classcolor, thickness=-1)
            # else:
            #     print("The mask shape is not supported")

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
        output_path = self.ui.outputPath.text()

        # save the generated mask
        filename = 'target_' + self.image_path.split("\\")[-1]
        plot_volume_orthoslices(self.dwidget.lmap, output_path + 'orthoslices_target_spheres.png')
        write_mrc(self.dwidget.lmap, output_path + filename)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    set_theme_style(app)


    application = MaskGenerationwindow()
    application.show()
    sys.exit(app.exec_())

# self.graphLayout = pg.GraphicsLayoutWidget()
# self.graphView = self.graphLayout.addViewBox()
# self.image_holder = pg.ImageItem()
# self.graphView.addItem(self.image_holder)
# self.graphView.setRange(QtCore.QRectF(0, 0, 512, 512))
# self.horizontalLayout_11.addWidget(self.graphLayout)