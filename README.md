# DRAW

paper: [A Recurrent Neural Network For Image Generation](https://arxiv.org/pdf/1502.04623.pdf)
reference: [https://github.com/ericjang/draw](https://github.com/ericjang/draw)

## Requirements

	git clone https://github.com/CosmosShadow/MLPythonLib

add MLPythonLib/lib to your python path like below

	# Mac
	echo 'export PYTHONPATH="path_to_where_clone/MLPythonLib/lib:$PYTHONPATH"' >> ~/.bash_profile

	# Ubuntu
	echo 'export PYTHONPATH="path_to_where_clone/MLPythonLib/lib:$PYTHONPATH"' >> ~/.bashrc


## Usage

	# train
	python train.py

	# generate images
	python gen.py