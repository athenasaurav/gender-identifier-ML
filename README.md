# Gender-Identifier-ML
Deep learning for gender identification from facial images written in keras and opencv

<h2>Sample output</h2>

![](output.jpg)

<h2>Usage</h2>

<p>There are two options to execute - either train the model from scratch or use the pre-trained VGGnet directly.</p>

<h3>If you do not want to train the model.</h3>
<p>pip install -r requirements.txt</p>
<p>python main.py -i input_image</p>
  
<p>You need matplotlib and scikit learn for training.</p>

<h3>To train the model -</h3>

<p>python train.py -d path-to-dataset</p>  

<p>This model dosen't genearalize well although the accuracy is 96% which is good enough for most ML pipelines. Feel free to tweak with  the hyperparameters or the architecture to achieve better results.</p>

<h2>TODO</h2>
<p>Make this work for image3.jpg and image4.jpg.</p>

<h2>NOTE</h2>

<p>I have merely created a wrapper. The original work can be found <a href="https://github.com/arunponnusamy/gender-detection-keras">here</a>. Thanks Arun for this great work.</p>



