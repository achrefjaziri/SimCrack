declare -a arr=("")

for batch in 32 #8 4
    do
      for set in  "${arr[@]}"
        do
          echo "$set"
          echo $batch

          ##python main.py  --arch_name  'cons_unet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 2.25 --att_connection --fuse_predictions
          #python main.py  --arch_name  'cons_unet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 2.25 --att_connection --fuse_predictions
          #python main.py  --arch_name  'cons_unet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 2.25 --att_connection --cons_loss --fuse_predictions
          #python main.py  --port 6016 --arch_name  'cons_unet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 2.25  --cons_loss --fuse_predictions
          #python main.py  --port 6016 --arch_name  'cons_unet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 2.25  --no-cons_loss --fuse_predictions
          #python main.py  --port 6017 --arch_name  'cons_unet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 2.25  --cons_loss --fuse_predictions --adain
          python main.py  --port 6017 --arch_name  'cons_unet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 2.25  --fuse_predictions --adain
          python main.py  --port 6017 --arch_name  'cons_unet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 2.25   --adain
          python main.py  --port 6017 --arch_name  'cons_unet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 2.25  --cons_loss --fuse_predictions
          python main.py  --port 6017 --arch_name  'cons_unet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 2.25   --fuse_predictions

          #python main.py  --arch_name  'cons_unet' --dataset  'SimResist' --set_size $set --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 1.75 --att_connection --fuse_predictions
          #python main.py  --arch_name  'cons_unet' --dataset  'SimResist' --set_size $set --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 1.75 --att_connection --cons_loss --fuse_predictions
          #python main.py  --arch_name  'cons_unet' --dataset  'SimResist' --set_size $set --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 1.75  --cons_loss --fuse_predictions
        done
    done