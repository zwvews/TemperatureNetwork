# Temperature Net in PyTorch

We provide a PyTorch implementation of Temperature Net. The code contains some unnecessary test codes and will be re-organized soon. 


## Prerequisites
- Linux
- Python 3
- Pytorch 1.0
- GPU + CUDA CuDNN


### Datasets
- [miniImageNet](https://drive.google.com/file/d/1fUBrpv8iutYwdL4xE1rX_R9ef6tyncX9/view).
- [StanfordDog](http://vision.stanford.edu/aditya86/ImageNetDogs/).
- [StanfordCar](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).



### Few-shot Classification
- For your convenience, we specify the parameters for every dataset, including the dataset path, # testing episodes, and # episodes to tune temperature (n*5/3). Please put the dataset in corresponding directory and directly run the corresponding file.
- Validation and test will be conducted while training.
- miniImageNet is splited as Ravi, Sachin, and Hugo Larochelle. "Optimization as a model for few-shot learning." (2016). 
- Train a 5-way 1-shot model for miniImageNet:
```bash
python Train_5way1shot_miniImageNet.py 
```
- Train a 5-way 5-shot model for miniImageNet:
```bash
python Train_5way5shot_miniImageNet.py 
```
- Train a 5-way 1-shot model for StanfordDog:
```bash
python Train_5way1shot_StanfordDog.py 
```
- Train a 5-way 5-shot model for StanfordDog:
```bash
python Train_5way5shot_StanfordDog.py 
```
- Train a 5-way 1-shot model for StanfordCar:
```bash
python Train_5way1shot_StanfordCar.py 
```
- Train a 5-way 5-shot model for StanfordCar:
```bash
python Train_5way5shot_StanfordCar.py 
```







