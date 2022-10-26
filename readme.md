

## Training Models ##
```
python main.py 
--arch_name  unet --dataset  SimCrack --batch_size 16 --resize_input --input_size 256
```
To Train Models with PMI calculations. It is preferable to use pre-calculated PMI maps to speed up training. For this purpose use the create_pmi_weights.py script.
```
python3 create_pmi_weights.py --phi_value 1.25 --no-resize_input --resize_size 512 --input_size 512 --input_ch 3
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
--no-resize_input --dataset SimResist
```


