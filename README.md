# Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning

This project is under submission to ACL 2023.

## To replicate our experiment result, you can follow the following steps.
1. Install the all required packages from the requirements_latest.txt `pip install -r requirements_latest.txt`

## Logical equivalence-driven data augmentation
### Synthetic sentences generation
1. You can run `logical_equivalence_synthetic_dataset.py` to automatically generate sentences which is ready for the stage-1 finetuning.
2. All code about logical equivalence data augmentation can be found in logical_equivalence_functions.py. You can run the script by `python logical_equivalence_functions.py`
3. To adjust the porprotion of positive and negative samples in the stage-1 finetuning, you can run the `negative_sample_extention.py`.
 
 ## Logical equivalence-driven data augmentation for representation learning
 You can follow the running script `script_running_notes.txt` and use the training commands to conduct stage-1 finetuning and stage-2 finetuning. Please remember you need to conduct the stage-1 finetuning firstly and then conduct the stage-2 finetuning. The main function code is in `BERT/run_glue_no_trainer.py`.
 Here is an example of stage-1 finetuning.
 
 Here is an example of stage-2 finetuning on MNLI.
 
 
 For the stage-2 finetuning on ReClor and LogiQA, you need to run the commands under the `BERT/scripts`. 
 Here is an example of stage-2 finetuning for ReClor.
 
 Here is an example of stage-2 finetuning for LogiQA.
