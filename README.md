A Modified version of 3D-UCaps

# Volumetric Macromolecule Identification in Cryo-Electron Tomograms using 3D Capsule Networks (3D-UCaps)

we employ capsule-based architecture to automate the task of macromolecule identification, that we refer to as 3D-UCaps. 3D-UCaps is a voxel-based Capsule network for medical image segmentation. In particular, the architecture is composed of three components: feature extractor, capsule encoder, and CNN decoder. The feature extractor converts voxel intensities of input sub-tomograms to activities of local features. The encoder is a 3D Capsule Network (CapsNet) that takes local features to generate a low-dimensional representation of the input. Then, a 3D CNN decoder reconstructs the sub-tomograms from the given representation by upsampling.

![alt text](imgs/NetDiagram.png "UCaps architecture")

Details of the 3D-UCaps model architecture can be found here [following paper](https://rdcu.be/cyhMv):

```
@inproceedings{nguyen20213d,
  title={3D-UCaps: 3D Capsules Unet for Volumetric Image Segmentation},
  author={Nguyen, Tan and Hua, Binh-Son and Le, Ngan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={548--558},
  year={2021},
  organization={Springer}
}
```

### Installation
We provide instructions on how to install dependencies via conda. First, clone the repository locally:
```
git clone https://github.com/VinAIResearch/3D-UCaps.git
```

Then, install dependencies depends on your cuda version. We provide two versions for CUDA 10 and CUDA 11
```
conda env create -f environment_cuda11.yml
or
conda env create -f environment_cuda10.yml
```

### Data preparation
We expect the data directory structure to be as follows:

```
path/to/<dataset>/
  imagesTr
  labelsTr
```
=======
### Training

Arguments for training can be divided into 3 groups:
