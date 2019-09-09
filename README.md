## Structure of the repo:

- **data**
    
    Datasets and all data-related stuff is here 
    (could be symbolic link to the real data location)

- **experiments** 

    Folder contains 
    * [cfgs] param (config) files for experiments;
    * [logs] all intermediate results from different experiments (param files, checkpoints, logs, ...);
    * [scripts] scripts to run specific experiments; 
                
- **notebooks**

    Place for jupyter notebooks.
    Notebooks could contain some analysis
    (dataset analysis, evalution results), demo, some ongoing work

- **material**

    Data, which include additional images, results, model weights, ...

    * [results] useful results are stored here;
    * [images] images for demo, readme, ...;
    * [weights] weights of the models (e.g. pretrained backbones ) 
    
- **src**

    Codebase.

## Requirements

Tested with

* python 3.6
* pytorch 1.2, torchvision 0.4
* tensorboard from (tf-nightly-2.0-preview)
        
## How to use

* load the data

(penn_fudan) go to the folder 'data' and run [YOU NEED ~ 300 MB]
> ./penn_fudan.sh

* 'demo' (output will be stored in $ROOT$/materials/images)
> python demo.py

* 'train'

run inside folder $ROOT$/src
> python train.py --param_file ../experiments/cfgs/penn_train.yaml

* 'test'
> python test.py --param_file ../experiments/cfgs/penn_test.yaml
