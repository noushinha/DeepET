import os.path
import math
import os.path
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image as impil

base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/results/IOU"
image_path = os.path.join(base_dir, "3DUnet_Slide_102.png")

# loading the image
img = cv2.imread(image_path)
scale_percent = 50 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
print(img.shape)
# cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# drawing rectangles
FONT_SCALE = 4e-3  # Adjust for larger font size in all images
THICKNESS_SCALE = 0.004  # Adjust for larger thickness in all images
TEXT_Y_OFFSET_SCALE = 0.05
img = cv2.rectangle(img, (6260, 5615), (8100, 7455), color=(0, 0, 255), thickness=40)
cv2.putText(img, "1", (6260, 5615 - int(height * TEXT_Y_OFFSET_SCALE)),
            fontScale=min(width, height) * FONT_SCALE,
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, color=(0, 0, 255),
            thickness=math.ceil(min(width, height) * THICKNESS_SCALE))
cv2.rectangle(img, (3610, 4400), (4710, 5500), color=(0, 255, 255), thickness=40)
cv2.putText(img, "2", (3180, 4930 - int(height * TEXT_Y_OFFSET_SCALE)),
            fontScale=min(width, height) * FONT_SCALE,
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, color=(0, 255, 255),
            thickness=math.ceil(min(width, height) * THICKNESS_SCALE))
sub_reg1 = img[5615:7455, 6260:8100]
sub_reg2 = img[4400:5500, 3610:4710]

fig1, ax1 = plt.subplots(figsize=(10, 10))
cmap = mpl.colors.ListedColormap([(0, 0, 0), (0.004, 0.4, 0.3686), (0.9647, 0.9098, 0.7647), (0.7490, 0.5058, 0.1764)])

# fig2, ax2 = plt.subplots(figsize=(10, 10))
# ax2.imshow(sub_reg1[..., ::-1], cmap=cmap)
# fig3, ax3 = plt.subplots(figsize=(10, 10))
# ax3.imshow(sub_reg2[..., ::-1], cmap=cmap)

pos = ax1.imshow(img[..., ::-1], cmap=cmap)
plt.xticks([])
plt.yticks([])


# colorbar setting
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.07)
cbar = fig1.colorbar(pos, ax=ax1, cax=cax)
cbar.ax.set_yticklabels(['TN', 'TP', 'FP', 'FN'])
filename = os.path.join(base_dir, "3DUnet_Slide_output.png")
plt.savefig(filename, format='png', dpi=300, bbox_inches="tight")


plt.show()

#legend
# cbar = plt.colorbar(cmap)
# cbar.ax.set_yticklabels(['0', '1', '2', '>3'])
# cbar.set_label('# of contacts', rotation=270)
# bounds = ['$TN$', '$TP$', '$FP$', '$FN$']
# ax_cb2 = fig.add_axes((0.125, 0.95, 0.75, 0.03))
# cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=cmap,
#                                 ticks=bounds,
#                                 spacing='proportional',
#                                 orientation='horizontal'
#                                 )
# ax_cb1 = fig.add_axes((0.85, 0.125, 0.03, 0.75))
#
# black_white = mpl.colors.ListedColormap([(0, 0, 0), (0.004, 0.4, 0.3686), (0.9647, 0.9098, 0.7647), (0.7490, 0.5058, 0.1764)])
# cb1 = mpl.colorbar.ColorbarBase(ax_cb1, cmap=black_white, orientation='vertical')

# ax_cb2 = fig.add_axes((0.125, 0.95, 0.75, 0.03))
# BR_cdict = {
#     'red':    ((0.0, 0.0, 0.0),
#                (1.0, 1.0, 1.0)),
#      'green': ((0.0, 0.0, 0.0),
#                (1.0, 0.0, 0.0)),
#      'blue':  ((0.0, 1.0, 1.0),
#                (1.0, 0.0, 0.0))
#     }
# blue_red = mpl.colors.LinearSegmentedColormap('BlueRed', BR_cdict)
# norm = mpl.colors.Normalize(vmin=0, vmax=255)
# cb2 = mpl.colorbar.ColorbarBase(
#     ax_cb2, cmap=blue_red, norm=norm, orientation='horizontal'
# )
# plt.show()
