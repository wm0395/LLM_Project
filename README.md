Init
To start with the model, 

**1. Get Activations**
*!python get_activations.py --model_name vicuna_7B --dataset_name conan*

**2. Change Directory to validation**

**3. Run the following command**
*!python validate_2fold.py --model_name vicuna_7B --dataset_name conan --num_heads 48 --alpha 15 --device 1 --use_center_of_mass --activations_dataset conan*
