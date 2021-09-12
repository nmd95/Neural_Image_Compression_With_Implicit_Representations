# Neural Image Compression With Implicit Representations
This project deals with image compression using a FC net overfitted on the image, with addition of pruning and quantization. 

Full project report is in *add path to pdf once finished*

## Installation
First, clone the project to your local machine:
```
git clone https://github.com/nmd95/Neural_Image_Compression_With_Implicit_Representations.git
```

Next, make sure to install the dependencies. You can either install them to your global folder or create a conda environment.

**Install to your global folder**
```
pip install -r requirements.txt
```

**Create conda environment**
```
conda env create -f environment.yml
```

## Usage
In order to run ```compress.py``` you should run the following code:
```
python compress.py --img-path img_path --save-path output_path --pruning-rates-list rates_list --epochs-per-prune-list epochs_list --num-layers number_of_layers --layer-width layer_width
```
Where: 

* ```img-path``` is the path to the image you wish to compress
* ```save-path``` is the path to the location in which to save the output to
* ```pruning-rates-list``` is a list of pruning rates you'd like the net to perform in each iteration. i.e. if you wish the net to be pruned twice, for the first time with pruning ratio of 0.2, and for the second time with a pruning ratio of 0.3 (total pruning of  24%)
write: 
```--pruning-rates-list 0.2,0.3```
* ```epochs-per-prune-list``` is the number of epochs you wish to train before every pruning. i.e. if you wish to prune twice (as mentioned above), and you want to train once for 50 epochs, and after the first pruning to train 20 epochs more, write: ```--epochs-per-prune-list 50,20```
* ```num-layers``` is the number of layers of the FC
* ```layer-width``` is the width of each layer

### Notes: 
* All parameters required. There are not any default setting. 
* PyTorch implementation for pruning used a mask matrix, and therefore the weight of the net after pruning will double. This issue can be easily dealt by saving the weights matrix instead. 

## Authors and acknowledgment
This project was made by Nimrod De La Vega and Doron Antebi as part of a DL-workshop at Tel Aviv University.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


Please make sure to update tests as appropriate.
