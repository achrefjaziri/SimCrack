
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

