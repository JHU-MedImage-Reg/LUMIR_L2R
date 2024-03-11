# Large Scale Unsupervised Brain MRI Image Registration (LUMIR)
This official repository houses baseline methods, training scripts, and pretrained models for the LUMIR challenge at Learn2Reg 2024.\
The challenge is dedicated to ***unsupervised*** brain MRI image registration and offers a comprehensive dataset of over 4000 preprocessed T1-weighted 3D brain MRI images, available for training, testing, and validation purposes.

Please visit [https://learn2reg.grand-challenge.org/](https://learn2reg.grand-challenge.org/) for more information.

### Dataset: 
Training dataset
- ***Preprocessing:*** The OpenBHB dataset was first preprocessed, including skull-stripping and affine registration, by the dataset creators. For details, please see [OpenBHB GitHub](https://baobablab.github.io/bhb/dataset) page and their [publication](https://www.sciencedirect.com/science/article/pii/S1053811922007522). We then applied N4 bias correction using ITK and intensity normalization using a [pre-existing tool](https://github.com/jcreinhold/intensity-normalization). 
- ***Image size:*** Each image and label map has a size of `160 x 192 x 224`.
- ***Normalization:*** The intensity values of each image volume are normalized into a range `[0,1]`.
- ***Dataset structure:***
    ```bash
    LUMIR_data/Train/------
            subject_0001.nii.gz   <--- a brain T1 MR image
            subject_0002.nii.gz
            .......
    LUMIR_data/Val/------
            subject_000x.nii.gz
            subject_000x.nii.gz
            .......
    ```

### Baseline methods:
1. VFA
2. TransMorph
3. VoxelMorph
5. DeedsBCV

### Evaluation metrics:
1. TRE
2. Dice
3. Non-diffeomorphic volumes

### Pre-trained model weights
1. TransMorph


### Citations:
The OpenBHB dataset was initially curated for an independent study. Our team has preprocessed this dataset for use in a prior image registration study. The dataset is available under a CC license. Should you use this dataset in your research, we kindly request that you cite the following papers:

***OpenBHB:***

    @article{dufumier2022openbhb,
    title={Openbhb: a large-scale multi-site brain mri data-set for age prediction and debiasing},
    author={Dufumier, Benoit and Grigis, Antoine and Victor, Julie and Ambroise, Corentin and Frouin, Vincent and Duchesnay, Edouard},
    journal={NeuroImage},
    volume={263},
    pages={119637},
    year={2022},
    publisher={Elsevier}
    }

A portion of the evaluation dataset was sourced from the AFIDs-OASIS dataset. If this dataset was used in your evaluation, such as participating in the L2R challenge, please cite the following paper in your work:

***AFIDs-OASIS:***

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

***OASIS:***
  
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
