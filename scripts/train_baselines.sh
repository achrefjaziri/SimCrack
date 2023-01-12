declare -a arr=(#"1000"
                #"2000" "4000"
                ''
                )


for batch in 16 32 #8 4
    do
      for set in "${arr[@]}"
        do
          echo "$set"
          python main.py --port 6016 --arch_name  'munet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1
          python main.py --port 6016 --arch_name  'unet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1
          python main.py --port 6016 --arch_name  'att_unet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1
          #python main.py --port 6014 --arch_name  'pmiunet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 1.75
          python main.py --port 6016 --arch_name  'pmiunet' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 2.25
          python main.py --port 6016 --arch_name  'munet_pmi' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 2.25
          #python main.py --port 6014 --arch_name  'munet_pmi' --dataset  'SimResist' --set_size "$set" --batch_size $batch --resize_crop_input --input_size 256 --input_ch 1 --histequalize_pmi --phi_value 1.75
        done
    done