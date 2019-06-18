# Gender-Identifier-ML
Deep learning for gender identification from facial images written in keras and opencv

## Sample output

![](output.jpg)

## Usage

There are two options to execute - either train the model from scratch or use the pre-trained VGGnet directly.

## If you do not want to train the model.

```
pip install -r requirements.txt
python main.py -i input_image
```
  
You need matplotlib and scikit learn for training.

## To train the model

```
python train.py -d path-to-dataset
```

This model dosen't genearalize well although the accuracy is 96% which is good enough for most ML pipelines. Feel free to tweak with  the hyperparameters or the architecture to achieve better results.

## TODO
Make this work for image3.jpg and image4.jpg.

## References

[Gender detection (from scratch) using deep learning with keras and cvlib](https://github.com/arunponnusamy/gender-detection-keras).
