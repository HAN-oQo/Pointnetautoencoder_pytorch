# Pointnet autoencoder pytorch

## Download_data

```
mkdir data && cd data
mkdir ModelNet && cd ModelNet
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1zE1d_eYD_QEnmS01LlZlEOMSZTIXRwIA" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1zE1d_eYD_QEnmS01LlZlEOMSZTIXRwIA" -o modelnet_classification.h5
md5sum modelnet_classification.h5

```
The output should be exactly the same as this:
```
87e763a66819066da670053a360889ed  modelnet_classification.h5
```
## Models
There are model.py and model1.py in model directory.  
Both have 128 dimensional latent space, and they have slightly different architecture.
Model1 have better performance.

I used the average Chamfer distance as Test Loss
  
## How to train
At train.ipynb,
```
# Set hyperparameters.
args = easydict.EasyDict({
    'train': True,
    'batch_size': 32,       # input batch size
    'n_epochs': 50,         # number of epochs
    'n_workers': 4,         # number of data loading workers
    'learning_rate': 0.001, # learning rate
    'beta1': 0.9,           # beta 1
    'beta2': 0.999,         # beta 2
    'step_size': 20,        # step size
    'gamma': 0.5,           # gamma
    'in_data_file': 'data/ModelNet/modelnet_classification.h5', # data directory
    'model': '',  # model path
    'model_type': 'hankyu1'             # hankyu = model, hankyu1 = model1
})
```
make sure train = true, and model = ''.
## How to eval
At train.ipynb,
```
# Set hyperparameters.
args = easydict.EasyDict({
    'train': False,
    'batch_size': 32,       # input batch size
    'n_epochs': 50,         # number of epochs
    'n_workers': 4,         # number of data loading workers
    'learning_rate': 0.001, # learning rate
    'beta1': 0.9,           # beta 1
    'beta2': 0.999,         # beta 2
    'step_size': 20,        # step size
    'gamma': 0.5,           # gamma
    'in_data_file': 'data/ModelNet/modelnet_classification.h5', # data directory
    'model': 'saved_models/autoencoder_50.pth',  # model path
    'model_type': 'hankyu1'             # hankyu = model, hankyu1 = model1
})
```
make sure train = False and model = 'model path'.
## How to show the reconstructions
Use show.ipynb .  
  
Change the 'in_data_file' and 'model' options.

## Results with 50 epochs
![pointnet-test](https://user-images.githubusercontent.com/35250512/112812400-0f4cb680-90b8-11eb-89ab-f35baf3bcfdd.PNG)
![Poinnet_train](https://user-images.githubusercontent.com/35250512/112812412-1247a700-90b8-11eb-8e49-8c22ca770738.PNG)

### Ground Truth & Reconstruction Results
Ground Truth            |  Reconstruction result
:-------------------------:|:-------------------------:
 ![ground_truth0](https://user-images.githubusercontent.com/35250512/112812734-6bafd600-90b8-11eb-92f6-735fa8580cd5.PNG) | ![result0](https://user-images.githubusercontent.com/35250512/112812778-75d1d480-90b8-11eb-8a17-0e7e8ecf3714.PNG)
![ground_truth1](https://user-images.githubusercontent.com/35250512/112812809-7cf8e280-90b8-11eb-8bb7-795e19688df6.PNG)  | ![result1](https://user-images.githubusercontent.com/35250512/112812835-85511d80-90b8-11eb-98a0-ce4d78132ef8.PNG)
![ground_truth2](https://user-images.githubusercontent.com/35250512/112812879-91d57600-90b8-11eb-818f-aa7c38adf45e.PNG)  | ![result2](https://user-images.githubusercontent.com/35250512/112812893-9568fd00-90b8-11eb-95c7-996305be98b5.PNG)
![ground_truth3](https://user-images.githubusercontent.com/35250512/112812913-98fc8400-90b8-11eb-8870-c54859876f70.PNG)  | ![result3](https://user-images.githubusercontent.com/35250512/112812920-9c900b00-90b8-11eb-9aa0-3b313dacad4b.PNG)




## Helped by
[https://github.com/dhirajsuvarna/pointnet-autoencoder-pytorch/blob/master/infer.py](https://github.com/dhirajsuvarna/pointnet-autoencoder-pytorch/blob/master/infer.py)

https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh

[https://github.com/charlesq34/pointnet-autoencoder/blob/master/utils/show3d_balls.py](https://github.com/charlesq34/pointnet-autoencoder/blob/master/utils/show3d_balls.py)
