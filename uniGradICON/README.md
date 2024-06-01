
# uniGradICON for LUMIR challenge at Learn2Reg 2024

Steps to run this script:
1. Install uniGradICON (Please follow instructions listed [here](https://github.com/uncbiag/uniGradICON))
2. In the same virtual environment, install nibabel with "pip install nibabel"
3. Run the evaluation script with "python l2r_LUMIR_eval.py --weights_path=[weight_file_path] --data_folder=[LUMIR_data_folder] --io_steps=0 --device=0"
4. When you see the prompt in the terminal, type the folder name of the experiment. The results will be saved in ./evaluation_results
5. Find a file under ./evaluation_results/[experiment name]/submission/submission.zip

## Acknowledgement
Courtesy of [Lin Tian](https://github.com/lintian-a) at UNC Chapel Hill, who kindly wrote the script and provided the instructions.
