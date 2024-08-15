# Test phase Docker submission examples and templates for LUMIR challenge

For detailed instructions, please visit: `learn2reg.grand-challenge.org/lumir-test-phase-submission/`. Please note that this webpage is accessible only to participants.

**To successfully run the TransMorph example for building, exporting, and testing the Docker image, you need to download the pretrained weights (available [here](https://drive.google.com/uc?export=download&id=1SSqI88l1MdrPJgE4Rn8pqXnVfZNPxtry)) and place them in the `DockerImage_TransMorph/pretrained_weights/` directory.**

After placing the pretrained weights, follow these steps:
* Run `bash build.sh` to build the Docker image for the TransMorph inference script. [[code](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/main/Test_phase_submission/DockerImage_TransMorph/build.sh)]
* Run `bash export.sh` to export and save the Docker image as `reg_model.tar.gz`. [[code](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/main/Test_phase_submission/DockerImage_TransMorph/export.sh)]
* Run `bash test.sh` to test the built Docker image on the validation dataset. [[code](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/main/Test_phase_submission/DockerImage_TransMorph/test.sh)]
  * Make sure to update this section of the `test.sh` script: https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/8ce441d3f8110c8a68eeda2113bdeff977ce01f6/Test_phase_submission/DockerImage_TransMorph/test.sh#L8-L10 according to the paths for the `.json` dataset file (available [here](https://drive.google.com/uc?export=download&id=1b0hyH7ggjCysJG-VGvo38XVE8bFVRMxb)) and the input and output directories.

You can also test the pre-built Docker image provided by us. Simply download the Docker image [here](https://drive.google.com/uc?export=download&id=1DVipRZg9GVxQU67D_NgUkRDpQpNBMLnK) and run `bash test.sh` in the same directory.


Some useful commands to free your disk space if too many Docker images are built:\
`docker rmi -f $(docker images -aq) #delete all containers`\
`docker rm -vf $(docker ps -aq) #delete all containers including its volumes`\
`docker system prune #delete everything`
