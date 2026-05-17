PPT Materials - 2026-05-17

Files in this folder
- train_log.csv: actual training log exported by the training script
- training_curves.png: real curves generated from training
- loss_curve_legacy.png: older loss-only figure kept for reference

What to say in the PPT
- Training curve = training loss curve
- Validation curve in this project is not validation loss
- The model was validated on the EUVP validation set using no-reference metrics:
  UCIQE and UIQM
- Therefore, the correct validation curves are:
  1. Validation UCIQE vs epoch
  2. Validation UIQM vs epoch

Important note
- This repo did not save validation loss during training.
- In training/train.py, each epoch logs:
  epoch, train_loss, val_uciqe, val_uiqm, lr
- So a true validation-loss curve cannot be recovered from the existing logs.

Best honest presentation line
- "During training, loss was computed on paired EUVP training data, while validation was
  monitored on the EUVP validation set using no-reference underwater image quality metrics
  UCIQE and UIQM."

If the examiner asks for validation loss
- You can say the current pipeline uses no-reference validation because the EUVP validation
  folder in this project is treated as an input-only validation set.
- To get validation loss in future runs, the training code must be modified to compute the
  same combined loss on a paired validation loader and save it every epoch.
