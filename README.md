# Large Scale Unsupervised Brain MRI Image Registration (LUMIR)
[![Static Badge](https://img.shields.io/badge/MICCAI-SIG_BIR-%2337677e?style=flat&labelColor=%23ececec&link=https%3A%2F%2Fmiccai.org%2Findex.php%2Fspecial-interest-groups%2Fbir%2F)](https://miccai.org/index.php/special-interest-groups/bir/) ![Static Badge](https://img.shields.io/badge/MICCAI-Learn2Reg-%23214f5f?labelColor=%23ececec&link=https%3A%2F%2Flearn2reg.grand-challenge.org%2F) <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

This official repository houses baseline methods, training scripts, and pretrained models for the LUMIR challenge at Learn2Reg 2024.\
The challenge is dedicated to ***unsupervised*** brain MRI image registration and offers a comprehensive dataset of over 4000 preprocessed T1-weighted 3D brain MRI images, available for training, testing, and validation purposes.

Please visit ***learn2reg.grand-challenge.org*** for more information.

$${\color{red}New!}$$ - 10/07/2024 - Test phase ranking is available here [this section](#-test-phase-results), congrats to the winners!!\
08/14/2024 - Test phase submission is available, see [this section](#test-phase-submission-guidelines)!

## Dataset: 
- ***Download Training Dataset:*** Access the training dataset via Google Drive ([~52GB](https://drive.google.com/uc?export=download&id=1PTHAX9hZX7HBXXUGVvI1ar1LUf4aVbq9)).
- ***Sanity Check:*** Since LUMIR focuses on unsupervised image registration, segmentation labels and landmarks for both the training and validation datasets are kept private. However, we provide a small subset to enable participants to perform sanity checks before submitting their results to the Grand Challenge.
    - Segmentation labels for 5 images in the training dataset ([download](https://drive.google.com/uc?export=download&id=14IQ_hiyMoheQqB_LrveDayzFaOe0YrEP)) (*Note that these labels are provided solely for sanity-check purposes and should not be used for training. The segmentation used for the test images may differ from the ones provided here.*)
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
- Vector Field Attention ([Official website](https://github.com/yihao6/vfa))
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
|VFA| 0.7726 ± 0.0286 | 2.4949 | 0.0788 | 3.2127|
|TransMorph|0.7594 ± 0.0319|2.4225|0.3509|3.5074|
|uniGradICON (w/ IO)|0.7512 ± 0.0366|2.4514|0.0001|3.5080|
|uniGradICON (w/o IO)|0.7369 ± 0.0412|2.5733|0.0000|3.6102|
|SynthMorph|0.7243 ± 0.0294|2.6099|0.0000|3.5730|
|VoxelMorph|0.7186 ± 0.0340|3.1545|1.1836|3.9821|
|SyN (ATNs)|0.6988 ± 0.0561  |2.6497|0.0000|3.7048|
|deedsBCV|0.6977 ± 0.0274  |2.2230|0.0001|3.9540|
|Initial |0.5657 ± 0.0263  |4.3543|0.0000|4.7876|

## <img src="https://raw.githubusercontent.com/iampavangandhi/iampavangandhi/master/gifs/Hi.gif" width="30"> Test phase results:
|team|ISO*?|TRE↓ (mm)|Dice↑|HdDist95↓|NDV↓ (%)|Score|Rank|GitHub|
|---|---|---|---|---|---|---|---|---|
|honkamj                       |✗|<p align=center><sup>3.0878 ± 4.17</sup></p>|   <p align=center><sup>0.7851 ± 0.11</sup></p>|   <p align=center><sup>3.0352 ± 2.41</sup></p>|   <p align=center><sup>0.0025 ± 0.00</sup></p>|   0.814|1|[GitHub](https://github.com/honkamj/SITReg)|
|hnuzyx_next-gen-nn            |✗|<p align=center><sup>3.1245 ± 4.19</sup></p>|   <p align=center><sup>0.7773 ± 0.12</sup></p>|   <p align=center><sup>3.2781 ± 2.55</sup></p>|   <p align=center><sup>0.0001 ± 0.00</sup></p>|   0.781|2|-|
|lieweaver                     |✗|<p align=center><sup>3.0714 ± 4.22</sup></p>|   <p align=center><sup>0.7779 ± 0.12</sup></p>|   <p align=center><sup>3.2850 ± 2.64</sup></p>|   <p align=center><sup>0.0121 ± 0.00</sup></p>|   0.737|3|-|
|zhuoyuanw210                  |✗|<p align=center><sup>3.1435 ± 4.20</sup></p>|   <p align=center><sup>0.7726 ± 0.12</sup></p>|   <p align=center><sup>3.2331 ± 2.53</sup></p>|   <p align=center><sup>0.0045 ± 0.00</sup></p>|   0.723|4|-|
|LYU-zhouhu                    |✗|<p align=center><sup>3.1324 ± 4.21</sup></p>|   <p align=center><sup>0.7776 ± 0.12</sup></p>|   <p align=center><sup>3.2464 ± 2.53</sup></p>|   <p align=center><sup>0.0150 ± 0.00</sup></p>|   0.722|5|-|
|Tsubasa025                    |✓|<p align=center><sup>3.1144 ± 4.16</sup></p>|   <p align=center><sup>0.7701 ± 0.12</sup></p>|   <p align=center><sup>3.2555 ± 2.56</sup></p>|   <p align=center><sup>0.0030 ± 0.00</sup></p>|   0.702|6|
|uniGradICON (w/ ISO 50)              |✓|<p align=center><sup>3.1350 ± 4.18</sup></p>|   <p align=center><sup>0.7596 ± 0.13</sup></p>|   <p align=center><sup>3.4010 ± 2.63</sup></p>|   <p align=center><sup>0.0002 ± 0.00</sup></p>|   0.668|7|[GitHub](https://github.com/uncbiag/uniGradICON)|
|VFA                           |✗|<p align=center><sup>3.1377 ± 4.21</sup></p>|   <p align=center><sup>0.7767 ± 0.11</sup></p>|   <p align=center><sup>3.1505 ± 2.47</sup></p>|   <p align=center><sup>0.0704 ± 0.05</sup></p>|   0.667|8|[GitHub](https://github.com/yihao6/vfa)|
|lukasf                        |✗|<p align=center><sup>3.1440 ± 4.20</sup></p>|   <p align=center><sup>0.7639 ± 0.12</sup></p>|   <p align=center><sup>3.4217 ± 2.60</sup></p>|   <p align=center><sup>0.2761 ± 0.08</sup></p>|   0.561|9|-|
|Bailiang                      |✗|<p align=center><sup>3.1559 ± 4.16</sup></p>|   <p align=center><sup>0.7735 ± 0.12</sup></p>|   <p align=center><sup>3.3287 ± 2.57</sup></p>|   <p align=center><sup>0.0222 ± 0.01</sup></p>|   0.526|10|[GitHub](https://github.com/BailiangJ/rethink-reg)|
|TransMorph                    |✗|<p align=center><sup>3.1420 ± 4.22</sup></p>|   <p align=center><sup>0.7624 ± 0.12</sup></p>|   <p align=center><sup>3.4617 ± 2.67</sup></p>|   <p align=center><sup>0.3621 ± 0.09</sup></p>|   0.518|11|[GitHub](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)|
|TimH                          |✗|<p align=center><sup>3.1926 ± 4.17</sup></p>|   <p align=center><sup>0.7303 ± 0.13</sup></p>|   <p align=center><sup>3.5695 ± 2.61</sup></p>|   <p align=center><sup>0.0000 ± 0.00</sup></p>|   0.487|12|-|
|deedsBCV                      |✓|<p align=center><sup>3.1042 ± 4.20</sup></p>|   <p align=center><sup>0.6958 ± 0.14</sup></p>|   <p align=center><sup>3.9446 ± 2.71</sup></p>|   <p align=center><sup>0.0002 ± 0.00</sup></p>|   0.423|13|[GitHub](https://github.com/mattiaspaul/deedsBCV)|
|uniGradICON                   |✗|<p align=center><sup>3.2400 ± 4.21</sup></p>|   <p align=center><sup>0.7422 ± 0.13</sup></p>|   <p align=center><sup>3.5747 ± 2.66</sup></p>|   <p align=center><sup>0.0001 ± 0.00</sup></p>|   0.402|14|[GitHub](https://github.com/uncbiag/uniGradICON)|
|kimjin2510                    |✓|<p align=center><sup>3.2354 ± 4.28</sup></p>|   <p align=center><sup>0.7355 ± 0.14</sup></p>|   <p align=center><sup>3.7328 ± 2.69</sup></p>|   <p align=center><sup>0.0033 ± 0.01</sup></p>|   0.384|15|-|
|HongyuLyu                     |✗|<p align=center><sup>3.1962 ± 4.26</sup></p>|   <p align=center><sup>0.7596 ± 0.12</sup></p>|   <p align=center><sup>3.5511 ± 2.58</sup></p>|   <p align=center><sup>1.1646 ± 0.29</sup></p>|   0.379|16|-|
|SynthMorph                    |✗|<p align=center><sup>3.2276 ± 4.24</sup></p>|   <p align=center><sup>0.7216 ± 0.14</sup></p>|   <p align=center><sup>3.6136 ± 2.61</sup></p>|   <p align=center><sup>0.0000 ± 0.00</sup></p>|   0.361|17|[GitHub](https://martinos.org/malte/synthmorph/)|
|TS_UKE                        |✓|<p align=center><sup>3.2250 ± 4.23</sup></p>|   <p align=center><sup>0.7603 ± 0.12</sup></p>|   <p align=center><sup>3.6297 ± 2.78</sup></p>|   <p align=center><sup>0.0475 ± 0.03</sup></p>|   0.351|18|-|
|ANTsSyN                       |✓|<p align=center><sup>3.4845 ± 4.24</sup></p>|   <p align=center><sup>0.7025 ± 0.14</sup></p>|   <p align=center><sup>3.6877 ± 2.60</sup></p>|   <p align=center><sup>0.0000 ± 0.00</sup></p>|   0.265|19|[GitHub](https://github.com/ANTsX/ANTsPy)|
|VoxelMorph                    |✗|<p align=center><sup>3.5282 ± 4.32</sup></p>|   <p align=center><sup>0.7144 ± 0.14</sup></p>|   <p align=center><sup>4.0718 ± 2.79</sup></p>|   <p align=center><sup>1.2167 ± 0.27</sup></p>|   0.157|20|[GitHub](https://github.com/voxelmorph/voxelmorph)|
|ZeroDisplacement              |✗|<p align=center><sup>4.3841 ± 4.33</sup></p>|   <p align=center><sup>0.5549 ± 0.17</sup></p>|   <p align=center><sup>4.9148 ± 2.50</sup></p>|   <p align=center><sup>0.0000 ± 0.00</sup></p>|   0.157|20|-|

*: ***ISO*** *stands for Instance-specific Optimization*
## Evaluation metrics:
1. TRE ([Code](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/2e98e0f936d2806ba2e40cbbd78a36219e4f9610/L2R_LUMIR_Eval/evaluation.py#L169-L197))
2. Dice ([Code](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/2e98e0f936d2806ba2e40cbbd78a36219e4f9610/L2R_LUMIR_Eval/evaluation.py#L155-L159))
3. HD95 ([Code](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/2e98e0f936d2806ba2e40cbbd78a36219e4f9610/L2R_LUMIR_Eval/evaluation.py#L162-L166))
4. **Non-diffeomorphic volumes (NDV)** ([Code](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/c19670ba91f1cffb33bdfff040daa42bfbf72058/L2R_LUMIR_Eval/evaluation.py#L139-L154)) *See this [article](https://link.springer.com/article/10.1007/s11263-024-02047-1) published in IJCV, and its associated [GitHub papge](https://github.com/yihao6/digital_diffeomorphism)* 

## Test Phase Submission Guidelines:
The test set consists of 590 images, making it impractical to distribute and collect the deformation fields. As a result, the test set will not be made available to challenge participants. Instead, **participants are required to containerize their methods with Docker and submit their Docker containers for evaluation.** Your code won't be shared and will be only used internally by the Learn2Reg organizers.

Docker allows for running an algorithm in an isolated environment called a container.  In particular, this container will locally replicate your pipeline requirements and execute your inference script.

Detailed instructions on how to build your Docker container is availble at `learn2reg.grand-challenge.org/lumir-test-phase-submission/`\
**We have provided examples and templates for creating a Docker image for submission on our [GitHub](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/tree/main/Test_phase_submission). You may find it helpful to start with the example Docker submission we created for TransMorph (available [here](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/tree/main/Test_phase_submission/DockerImage_TransMorph)), or you can start from a blank template (available [here](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/tree/main/Test_phase_submission/DockerImage_Template)).**

Your submission should be a single `.zip` file containing the following things:
```bash
LUMIR_[your Grand Challenge username]_TestPhase.zip
└── [your docker image name].tar.gz <------ #Your Docker container
└── README.txt                      <------ #A description of the requirements for running your model, including the number of CPUs, amount of RAM, and the estimated computation time per subject.
└── validation_predictions.zip      <------ #A .zip file containing the predicted displacement fields for the validation dataset, ensuring the format adheres to ones outlined at this page.
    ├── disp_3455_3454.nii.gz
    ├── disp_3456_3455.nii.gz
    ├── disp_3457_3456.nii.gz
    ├── disp_3458_3457.nii.gz
    ├── ...
    └── ...
```
You will need to submit by **31st August 2024**

Please choose **ONE** of the following:
* **EITHER** Email the download link for your `.zip` file to **jchen245 [at] jhmi.edu**
* **OR** Upload your `.zip` file [here](https://cloud.imi.uni-luebeck.de/s/TFFaXnBzZTqtpx2).

## Validation Submission guidelines:
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

