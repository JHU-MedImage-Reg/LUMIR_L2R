# Large Scale Unsupervised Brain MRI Image Registration (LUMIR)
This official repository houses baseline methods, training scripts, and pretrained models for the LUMIR challenge at Learn2Reg 2024.\
The challenge is dedicated to ***unsupervised*** brain MRI image registration and offers a comprehensive dataset of over 4000 preprocessed T1-weighted 3D brain MRI images, available for training, testing, and validation purposes.

Please visit ***learn2reg.grand-challenge.org*** for more information.

## Dataset: 
- ***Download Training Dataset:*** Access the training dataset via Google Drive ([~52GB](https://drive.google.com/uc?export=download&id=1PTHAX9hZX7HBXXUGVvI1ar1LUf4aVbq9)).
- ***Sanity Check:*** Since LUMIR focuses on unsupervised image registration, segmentation labels and landmarks for both the training and validation datasets are kept private. However, we provide a small subset to enable participants to perform sanity checks before submitting their results to the Grand Challenge.
    - Segmentation labels for 5 images in the training dataset ([download](https://drive.google.com/uc?export=download&id=14IQ_hiyMoheQqB_LrveDayzFaOe0YrEP)) (*Note that these labels are provided solely for sanity-check purposes and should not be used for training. The segmentation used for the test images may differ from the ones provided here.*)
    - Manual landmarks for 10 images in the training dataset (TBA)
- ***Preprocessing:*** The OpenBHB dataset underwent initial preprocessing by its creators, which included skull stripping and affine registration. For comprehensive details, refer to the [OpenBHB GitHub](https://baobablab.github.io/bhb/dataset) page and their [article](https://www.sciencedirect.com/science/article/pii/S1053811922007522). Subsequently, we performed N4 bias correction with ITK and intensity normalization using a [pre-existing tool](https://github.com/jcreinhold/intensity-normalization).
- ***Annotation:*** We conducted segmentation of the anatomical structures using automated software. To enhance the dataset for evaluation purposes, an experienced radiologist and neurologist contributed manual landmark annotations to a subset of the images.
- ***Image size:*** The dimensions of each image are `160 x 224 x 192`.
- ***Normalization:*** Intensity values for each image volume have been normalized to fall within the range `[0,255]`.
- ***Dataset structure:***
    ```bash
    LUMIR/imagesTr/------
            LUMIRMRI_0000_0000.nii.gz   <--- a single brain T1 MR image
            LUMIRMRI_0001_0000.nii.gz
            LUMIRMRI_0002_0000.nii.gz
            .......
    LUMIR/imagesVal/------
            LUMIRMRI_3454_0000.nii.gz
            LUMIRMRI_3455_0000.nii.gz
    ```
- ***Dataset json file:*** [LUMIR_dataset.json](https://drive.google.com/uc?export=download&id=1b0hyH7ggjCysJG-VGvo38XVE8bFVRMxb)

## Baseline methods:
***Learning-based models:***
- VFA
- TransMorph ([Official website](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) | [Scripts](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/tree/main/TransMorph)  | [Pretrained weights (~355MB)](https://drive.google.com/uc?export=download&id=1SSqI88l1MdrPJgE4Rn8pqXnVfZNPxtry))
- VoxelMorph ([Official website](https://github.com/voxelmorph/voxelmorph) | [Scripts](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/tree/main/VoxelMorph)  | [Pretrained weights (~83MB)](https://drive.google.com/uc?export=download&id=1imUkWtf_15Ih2rxPTKfwuIP04eKr9S9H))

***Learning-based foundation models:***
- SynthMorph ([Official website](https://martinos.org/malte/synthmorph/) | [Scripts](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/tree/main/SynthMorph))
- UniGradICON ([Official website](https://github.com/uncbiag/uniGradICON) | [Scripts](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/tree/main/uniGradICON))
- BrainMorph ([Official website](https://github.com/alanqrwang/brainmorph) | [Scripts](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/tree/main/BrainMorph))

***Optimization-based methods:***
- SyN (ATNs) ([Official website](https://github.com/ANTsX/ANTsPy)) | [Scripts](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/tree/main/SyN%20(ATNs)))
- deedsBCV ([Official website](https://github.com/mattiaspaul/deedsBCV) | [Scripts](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/tree/main/deedsBCV))

***Validation dataset results for baseline methods***
|Model|Dice↑|TRE↓ (mm)|NDV↓ (%)|HdDist95↓|
|---|---|---|---|---|
|TransMorph|0.7594 ± 0.0319|2.4225|0.3509|3.5074|
|uniGradICON (w/ IO)|0.7512 ± 0.0366|2.4514|0.0001|3.5080|
|uniGradICON (w/o IO)|0.7369 ± 0.0412|2.5733|0.0000|3.6102|
|SynthMorph|0.7243 ± 0.0294|2.6099|0.0000|3.5730|
|VoxelMorph|0.7186 ± 0.0340|3.1545|1.1836|3.9821|
|SyN (ATNs)|0.6988 ± 0.0561  |2.6497|0.0000|3.7048|
|deedsBCV|0.6977 ± 0.0274  |2.2230|0.0001|3.9540|
|Initial |0.5657 ± 0.0263  |4.3543|0.0000|4.7876|

## Evaluation metrics:
1. TRE ([Code](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/2e98e0f936d2806ba2e40cbbd78a36219e4f9610/L2R_LUMIR_Eval/evaluation.py#L169-L197))
2. Dice ([Code](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/2e98e0f936d2806ba2e40cbbd78a36219e4f9610/L2R_LUMIR_Eval/evaluation.py#L155-L159))
3. HD95 ([Code](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/2e98e0f936d2806ba2e40cbbd78a36219e4f9610/L2R_LUMIR_Eval/evaluation.py#L162-L166))
4. **Non-diffeomorphic volumes (NDV)** ([Code](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/c19670ba91f1cffb33bdfff040daa42bfbf72058/L2R_LUMIR_Eval/evaluation.py#L139-L154)) *See this [article](https://link.springer.com/article/10.1007/s11263-024-02047-1) published in IJCV, and its associated [GitHub papge](https://github.com/yihao6/digital_diffeomorphism)* 



## Submission guidelines:
We expect to provide displacement fields for all registrations in the file naming format should be `disp_PatID1_PatID2`, where `PatID1` and `PatID2` represent the subject IDs for the fixed and moving images, respectively. The evaluation process requires the files to be organized in the following structure:
```bash
folder.zip
└── folder
    ├── disp_3455_3454.nii.gz
    ├── disp_3456_3455.nii.gz
    ├── disp_3457_3456.nii.gz
    ├── disp_3458_3457.nii.gz
    ├── ...
    └── ...
```
Submissions must be uploaded as zip file containing displacement fields (displacements only) for all validation pairs for all tasks (even when only participating in a subset of the tasks, in that case submit deformation fields of zeroes for all remaining tasks). You can find the validation pairs for in the LUMIR_dataset.json. The convention used for displacement fields depends on scipy's `map_coordinates()` function, expecting displacement fields in the format `[X, Y, Z,[x, y, z]]` or `[[x, y, z],X, Y, Z]`, where `X, Y, Z` and `x, y, z` represent voxel displacements and image dimensions, respectively. The evaluation script expects `.nii.gz` files using full-precision format and having shapes `160x224x196x3`. Further information can be found here.

Note for PyTorch users: When using PyTorch as deep learning framework you are most likely to transform your images with the `grid_sample()` routine. Please be aware that this function uses a different convention than ours, expecting displacement fields in the format `[X, Y, Z,[x, y, z]]` and normalized coordinates between -1 and 1. Prior to your submission you should therefore convert your displacement fields to match our convention.

## Citations for dataset usage:

    @article{dufumier2022openbhb,
    title={Openbhb: a large-scale multi-site brain mri data-set for age prediction and debiasing},
    author={Dufumier, Benoit and Grigis, Antoine and Victor, Julie and Ambroise, Corentin and Frouin, Vincent and Duchesnay, Edouard},
    journal={NeuroImage},
    volume={263},
    pages={119637},
    year={2022},
    publisher={Elsevier}
    }

    @article{taha2023magnetic,
    title={Magnetic resonance imaging datasets with anatomical fiducials for quality control and registration},
    author={Taha, Alaa and Gilmore, Greydon and Abbass, Mohamad and Kai, Jason and Kuehn, Tristan and Demarco, John and Gupta, Geetika and Zajner, Chris and Cao, Daniel and Chevalier, Ryan and others},
    journal={Scientific Data},
    volume={10},
    number={1},
    pages={449},
    year={2023},
    publisher={Nature Publishing Group UK London}
    }
    
    @article{marcus2007open,
    title={Open Access Series of Imaging Studies (OASIS): cross-sectional MRI data in young, middle aged, nondemented, and demented older adults},
    author={Marcus, Daniel S and Wang, Tracy H and Parker, Jamie and Csernansky, John G and Morris, John C and Buckner, Randy L},
    journal={Journal of cognitive neuroscience},
    volume={19},
    number={9},
    pages={1498--1507},
    year={2007},
    publisher={MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info~…}
    }

If you have used **Non-diffeomorphic volumes** in the evaluation of the deformation regularity, please cite the following:

    @article{liu2024finite,
      title={On finite difference jacobian computation in deformable image registration},
      author={Liu, Yihao and Chen, Junyu and Wei, Shuwen and Carass, Aaron and Prince, Jerry},
      journal={International Journal of Computer Vision},
      pages={1--11},
      year={2024},
      publisher={Springer}
    }

