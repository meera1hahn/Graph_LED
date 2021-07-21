## Evaluate

1. Evaluate the lingunet-skip model described in the paper (https://arxiv.org/abs/2011.08277). Will predict over nodes of all floor and evaluates via geodesic distance. 
   ```
   python run.py \
    --evaluate \
    --eval_ckpt ../../data/models/lingunet-skip.pt \
    --run_name lingunet-skip
   ```

## Training

1. Train the lingunet-skip model described in the paper (https://arxiv.org/abs/2011.08277). Will predict over nodes of all floor and trains based on the metric of geodesic distance. 
   ```
   python run.py \
    --train \
    --model_save \
    --run_name base_model 
   ```
