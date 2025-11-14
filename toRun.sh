export nnUNet_preprocessed=/home/evabreznik/Skrivbord/MAIASTUFF/nnunet_data/nnUNet_preprocessed
export nnUNet_raw=/home/evabreznik/Skrivbord/MAIASTUFF/nnunet_data/nnUNet_raw
export nnUNet_results=/home/evabreznik/Skrivbord/MAIASTUFF/nnunet_data/nnUNet_results


# Move the trainer file to the nnUNetTrainer folder
cp nnUNetTrainerMultitaskCAS.py ~/Skrivbord/MultitaskCASeg/nnUNet/nnunetv2/training/nnUNetTrainer/variants/. #network_architecture/.
# Plan and preprocess the data as usual, then copy the class labels in the same folder
#nnUNetv2_plan_and_preprocess -t 666 --verify_dataset_integrity
#cp $nnUNet_raw/Task666_ASOCA/class_labels.csv $nnUNet_preprocessed/Task666_ASOCA/.
cp class_labels.csv ../MAIASTUFF/nnunet_data/nnUNet_preprocessed/.
# Dont forget to add the classifier arguments into the json plans! 
#this can be done with a function. I did it manually.

# Run training with the new trainer:
nnUNetv2_train 666 3d_fullres 0 -tr nnUNetTrainerMultiCAS_bottleneck -p nnUNetResEncUNetMPlans --npz &
nnUNetv2_train 666 3d_fullres 1 -tr nnUNetTrainerMultiCAS_bottleneck -p nnUNetResEncUNetMPlans --npz
wait
nnUNetv2_train 666 3d_fullres 2 -tr nnUNetTrainerMultiCAS_bottleneck -p nnUNetResEncUNetMPlans --npz &
nnUNetv2_train 666 3d_fullres 3 -tr nnUNetTrainerMultiCAS_bottleneck -p nnUNetResEncUNetMPlans --npz 
wait
nnUNetv2_train 666 3d_fullres 4 -tr nnUNetTrainerMultiCAS_bottleneck -p nnUNetResEncUNetMPlans --npz &
nnUNetv2_train 666 3d_lowres 0 -tr nnUNetTrainerMultiCAS_bottleneck -p nnUNetResEncUNetMPlans --npz 
wait

python gather_class_results.py class_labels.csv nnUNet_results/nnUNet/666/3d_fullres/nnUNetTrainerMultiCAS_bottleneck
nnUNetv2_train 666 3d_lowres 1 -tr nnUNetTrainerMultiCAS_bottleneck -p nnUNetResEncUNetMPlans --npz&
nnUNetv2_train 666 3d_lowres 2 -tr nnUNetTrainerMultiCAS_bottleneck -p nnUNetResEncUNetMPlans --npz
wait
nnUNetv2_train 666 3d_lowres 3 -tr nnUNetTrainerMultiCAS_bottleneck -p nnUNetResEncUNetMPlans --npz&
nnUNetv2_train 666 3d_lowres 4 -tr nnUNetTrainerMultiCAS_bottleneck -p nnUNetResEncUNetMPlans --npz
wait
python gather_class_results.py class_labels.csv nnUNet_results/nnUNet/666/3d_lowres/nnUNetTrainerMultiCAS_bottleneck
nnUNetv2_train 666 3d_fullres 0 -tr nnUNetTrainerMultiCAS_penultimate -p nnUNetResEncUNetMPlans --npz 6
nnUNetv2_train 666 3d_fullres 1 -tr nnUNetTrainerMultiCAS_penultimate -p nnUNetResEncUNetMPlans --npz 
wait
nnUNetv2_train 666 3d_fullres 2 -tr nnUNetTrainerMultiCAS_penultimate -p nnUNetResEncUNetMPlans --npz &
nnUNetv2_train 666 3d_fullres 3 -tr nnUNetTrainerMultiCAS_penultimate -p nnUNetResEncUNetMPlans --npz 
wait
nnUNetv2_train 666 3d_fullres 4 -tr nnUNetTrainerMultiCAS_penultimate -p nnUNetResEncUNetMPlans --npz
python gather_class_results.py class_labels.csv nnUNet_results/nnUNet/666/3d_fullres/nnUNetTrainerMultiCAS_penultimate