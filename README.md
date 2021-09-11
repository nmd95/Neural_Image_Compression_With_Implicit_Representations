# Neural Image Compression With Implicit Representations
This project deals with image compression using a FC net overfitted on the image, with addition of pruning and quantization. 

Full project report is in *add path to pdf once finished*

## Usage
In order to run ```compress.py``` you should run the following code:
```
python copmress.py --img-path img_path --save-path output_path --pruning-rates-list rates_list --epochs-per-prune-list epochs_list --model-arch model_architecture --arch-params arch_params_list
```
Where: 

* ```img-path``` is the path to the image you wish to compress
* ```save-path``` is the path to the location in which to save the output to
* ```pruning-rates-list``` is a list of pruning rates you'd like the net to perform in each iteration. i.e. if you wish the net to be pruned twice, for the first time with pruning ratio of 0.2, and for the second time with a pruning ratio of 0.3 (total pruning of  24%)
write: 
```--pruning-rates-list 0.2,0.3```
* ```epochs-per-prune-list``` is the number of epochs you wish to train before every pruning. i.e. if you wish to prune twice (as mentioned above), and you want to train once for 50 epochs, and after the first pruning to train 20 epochs more, write: ```--epochs-per-prune-list 50,20```
* ```model-arch``` is the architecture of the FC net. i.e. if you want to train a 10-layered network of depth 40, write: ```--model-arch 10_40```
* ```arch-params``` is the parameters of the FC net, receives a tuple of (number_of_layers, width of a layer). i.e. if you wrote ```10_40``` as the model_arch parameter, for arch-parames 
you need to write: ```--arch-params 10,40```

### Notes: 
* All parameters required. There are not any default setting. 
* PyTorch implementation for pruning used a mask matrix, and therefore the weight of the net after pruning will double. This issue can be easily dealt by saving the weights matrix instead. 

## Authors and acknowledgment
This project was made by Nimrod De La Vega and Doron Antebi as part of a DL-workshop at Tel Aviv University.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


Please make sure to update tests as appropriate.
