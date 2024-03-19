# Large Scale Unsupervised Brain MRI Image Registration (LUMIR)
This official repository houses baseline methods, training scripts, and pretrained models for the LUMIR challenge at Learn2Reg 2024.\
The challenge is dedicated to ***unsupervised*** brain MRI image registration and offers a comprehensive dataset of over 4000 preprocessed T1-weighted 3D brain MRI images, available for training, testing, and validation purposes.

Please visit [https://learn2reg.grand-challenge.org/](https://learn2reg.grand-challenge.org/) for more information.

## Dataset: 
- ***Download Training Dataset:*** Access the training dataset via Google Drive.
- ***Preprocessing:*** The OpenBHB dataset underwent initial preprocessing by its creators, which included skull stripping and affine registration. For comprehensive details, refer to the [OpenBHB GitHub](https://baobablab.github.io/bhb/dataset) page and their [article](https://www.sciencedirect.com/science/article/pii/S1053811922007522). Subsequently, we performed N4 bias correction with ITK and intensity normalization using a [pre-existing tool](https://github.com/jcreinhold/intensity-normalization).
- ***Annotation:*** We conducted segmentation of the anatomical structures using automated software. To enhance the dataset for evaluation purposes, an experienced radiologist and neurologist contributed manual landmark annotations to a subset of the images.
- ***Image size:*** The dimensions of each image are `160 x 192 x 224`.
- ***Normalization:*** Intensity values for each image volume have been normalized to fall within the range `[0,255]`.
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

## Baseline methods:
1. VFA
2. TransMorph
3. SynthMorph
4. VoxelMorph
5. DeedsBCV
6. UniGradICON

## Evaluation metrics:
1. TRE
2. Dice
3. **Non-diffeomorphic volumes** *See this [article](https://arxiv.org/abs/2212.06060) published in IJCV, and its associated [GitHub papge](https://github.com/yihao6/digital_diffeomorphism)*

## Pre-trained model weights
1. TransMorph
2. VFA
3. VoxelMorph


## Citations:
The OpenBHB dataset was initially curated for an independent study. Our team has preprocessed this dataset for use in this challenge. The dataset is available under a CC license. Should you use this dataset in your research, we kindly request that you cite the following paper:

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

If you have used **Non-diffeomorphic volumes** in the evaluation of the deformation regularity, please cite the following:

    @article{liu2022finite,
    title={On finite difference jacobian computation in deformable image registration},
    author={Liu, Yihao and Chen, Junyu and Wei, Shuwen and Carass, Aaron and Prince, Jerry},
    journal={arXiv preprint arXiv:2212.06060},
    year={2022}
    }
