

## Training Models ##
```
python main.py  --arch_name  unet --dataset  SimResist --batch_size 8 --resize_crop_input --input_size 256 --input_ch 1
```

Train Munet-pmi
```
python main.py --arch_name munet_pmi --dataset SimResist --batch_size 16 --resize_input --input_size 256 --input_ch 1 --histequalize_pmi --num_epochs 100
```

To Train Models with PMI calculations. It is preferable to use pre-calculated PMI maps to speed up training. For this purpose use the create_pmi_weights.py script.
```
python3 create_pmi_weights.py --phi_value 1.25 --no-resize_crop_input --resize_size 512 --input_size 512 --input_ch 3
```

## Evaluation ##

Create predictions maps
```
python run_segmentation.py 
--test_mode
--arch_name  pmiunet
--model_path  /home/ajaziri/resist_projects/SimCrack/workspace/pmiunet/SimResist/2022-10-04_10-33-31
--input_ch  3 --input_size  256  --patch_size  512  --patchwise_eval --dataset  RealResist
```

Create predictions on Simulated Data
```
python run_segmentation.py 
--test_mode
--arch_name  pmiunet
--model_path  /home/ajaziri/resist_projects/SimCrack/workspace/pmiunet/SimResist/2022-10-04_10-33-31
--input_ch  3 --input_size 256
--no-resize_crop_input --dataset SimResist
```


Full Pipeline Evaluation
```
python run_full_evaluation_pipeline.py 
--dataset RealResist
--gt_path  path/to/ground_truth_masks
--patch_wise_eval 
```

python run_full_evaluation_pipeline.py  --dataset RealResist  --gt_path  /data/resist_data/datasets/resist_set/gts  --patchwise_eval