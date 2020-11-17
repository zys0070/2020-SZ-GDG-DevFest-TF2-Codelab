
# There are 3 types of TF2 mnist model construction and training scripts.
# If you can't wait to train models, one pre-trained model is provided.
# ov_py_mnist_example folder includes python sample and test images.


# Installation
Both the TF2 and OpenVINO tool are needed to install first.

# Convert TF2 model to OpenVINO intermediate representation (IR)
mo_tf.py --saved_model_dir <model_path> --input_shape [1,28,28,1] --reverse_input_channels

# Run Mnist sample
python inference.py -m <path to .xml> -i <path to image>

