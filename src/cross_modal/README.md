## Evaluate

1. Evaluate the crossmodal simple model -- run with nn.DataParallel(self.model)
   ```
   python run.py --evaluate --name crossmodal_simple --eval_ckpt ../../data/models/crossmodal_simple.pt
   ```

2. Evaluate the crossmodal w/attention model
   ```
   python run.py --attention --evaluate --name crossmodal_attention --eval_ckpt ../../data/models/crossmodal_att.pt
   ```

## Training

1. Train the crossmodal simple model
   ```
   python run.py --train --name crossmodal_simple --model_save
   ```

2. Train the crossmodal w/attention model
   ```
   python run.py --attention --train --name crossmodal_attention --model_save
   ```