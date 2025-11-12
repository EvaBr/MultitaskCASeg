export nnUNet_preprocessed=/Users/evabreznik/Desktop/MAIASTUFF/nnunet_data/nnUNet_preprocessed
export nnUNet_raw=/Users/evabreznik/Desktop/MAIASTUFF/nnunet_data/nnUNet_raw
export nnUNet_results=/Users/evabreznik/Desktop/MAIASTUFF/nnunet_data/nnUNet_results


# Move the trainer file to the nnUNetTrainer folder
cp nnUNetTrainerMultitaskCAS.py ~/Desktop/KTH_CAS/MultitaskCASeg/nnUNet/training/nnUNetTrainer/variants/network_architecture/.

# Plan and preprocess the data as usual
#nnUNetv2_plan_and_preprocess -t 666 --verify_dataset_integrity

# Dont forget to add the classifier arguments into the json plans! 
#this can be done with a function. I did it manually.

# Run training with the new trainer:
nnUNetv2_train 666 3d_fullres 0 -tr nnUNetTrainerMultitaskCAS
nnUNetv2_train 666 3d_fullres 1 -tr nnUNetTrainerMultitaskCAS
nnUNetv2_train 666 3d_fullres 2 -tr nnUNetTrainerMultitaskCAS
nnUNetv2_train 666 3d_fullres 3 -tr nnUNetTrainerMultitaskCAS
nnUNetv2_train 666 3d_fullres 4 -tr nnUNetTrainerMultitaskCAS

