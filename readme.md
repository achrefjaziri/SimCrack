
# Designing a Hybrid Neural System to Learn Real-world Crack Segmentation from Fractal-based Simulation #


- Open-source code for CAP-Net, the semantic segmentation model presented in our paper "Designing a Hybrid Neural System to Learn Real-world Crack Segmentation from Fractal-based Simulation" (https://arxiv.org/abs/2309.09637v1)

- The code for the Cracktal Simulator can found in a second repository: https://github.com/achrefjaziri/CracktalConcreteSimulator/tree/master
## Dataset ##

- The annotated real images is available at:  https://zenodo.org/records/10071534
- The dataset of cracktal generated synthetic images is too large to host on Zenodo, please contact the first author at: Jaziri@em.uni-frankfurt.de to get the direct download link. 
- Please note that the dataset is licensed for non-commercial and educational use only as specified by the license file attached with the dataset at above link.

## Usage ##


### Requirements ###
- You can create an environment using  ```conda env create -f environment.yml```
. Note that most of the packages inside are not actually used, but provide a full list for full reproducibility. We don't foresee any problems with using newer versions of Pytorch to replicate out results.
- When setting up this project, we recommend you follow the following directory schemes to properly load the data for training and evaluation: 
- additionally to train our CAP-Net, we recommend you donwload the pre-computed PMI and style transfered maps from .. to speed up the training process.
- Further below, we provide code for generation of the style transfered maps as well as the PMI. 

### Training The Models ###

To replicate the results of our paper, we provide bash scripts to train various baselines mentionned in the paper as well as the CAP-Net model.
To run a simple U-Net model with Cracktal, you can use the following command:


```
python main.py  --arch_name  unet --dataset  SimResist --batch_size 32 --resize_crop_input --input_size 256 --input_ch 1
```

Train Munet-pmi
```
python main.py --arch_name munet_pmi --dataset SimResist --batch_size 16 --resize_input --input_size 256 --input_ch 1 --histequalize_pmi --num_epochs 100
```

To speed the training of models with PMI or style transfer, it is preferable to use pre-calculated maps. 

To generate the PMI maps, use the following command:
```
python3 create_pmi_weights.py --phi_value 1.25 --no-resize_crop_input --resize_size 512 --input_size 512 --input_ch 3
```
To generate the augmented Cracktal maps:
```
python3 style_transfer.py --no-resize_crop_input --resize_size 512 --input_size 512 --input_ch 3
```


### Evaluation ###

Generate predictions maps for 
```
python run_segmentation.py 
--test_mode
--arch_name  pmiunet
--model_path  /path/to/model
--input_ch  3 --input_size  256  --patch_size  512  --patchwise_eval --dataset  RealResist
```

Create predictions on Simulated Data
```
python run_segmentation.py 
--test_mode
--arch_name  pmiunet
--model_path  /path/to/model
--input_ch  3 --input_size 256
--no-resize_crop_input --dataset SimResist
```


Full Pipeline Evaluation
```
python run_full_evaluation_pipeline.py 
--dataset RealResist
--gt_path  path/to/ground_truth_masks
--patchwise_eval 
```


```
python run_full_evaluation_pipeline.py  --dataset RealResist  --gt_path  /data/resist_data/datasets/resist_set/gts  --patchwise_eval
```
