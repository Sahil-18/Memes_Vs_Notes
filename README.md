# Memes Vs Notes
This deep learning programme is a part of VNIT's IvLabs Summer Internship Programme. This programme is developed using two technique. 
1. Linear Neural Network using Numpy Library
2. Convolutional Neural Network using Pytorch Library

## Dataset Link 
https://drive.google.com/drive/folders/1qVxobdyzJ_N4G1RO5l8fnTG0u5NyGmwZ?usp=sharing

Above is the link for Memes Vs Notes Dataset which I used. It contains two folder. 
First is 'memes' which contains 800 memes and second is 'notes' which also contains 800 notes.
The size of photos is 64*64 in the .jpg format.
In case you want to use your dataset, you can do that by giving drive address of that data in required places in programme. 
This places are specified with some comments so that you can easily recognise them.

## About programme
The programme is defined two differentiate between a meme and a note. It predicts that a given input image is a meme or a note.
For this operation, first of all we have to train the model so that it can learn through it. This requires a massive data to train the model.
Therefore to serve the purpose, we need to split the data. Here I have splitted data into three parts:
#### 1. Training Set (75% of total dataset)
#### 2. Dev/Validation Set (12.5% of toatal dataset)
#### 3. Testing Set (12.5% of total dataset)
Training set is used to train the model, validation set is used to verify whether model is correctly learned and trained.
Test set is used to test the model and outputs accuracy of model which specifies how accurately model is differentiating between memes and notes.
We define the model by creating a Neural Network which can recognise image and learn from it. Here, I have used two methods to define a Neural Network.
First is using Numpy Library and other one is Convolutional Neural Network using Pytorch Library.


### Linear Neural Network using Numpy Library
In this method, each data image is converted to a Numpy array. Since these images are coloured scale i.e in the RGB form, we convert these image into a column vector (featur 
vector) of size [12288,1] since height and width of image is 64,64 respectively and image has 3 channels. These image column array is used to train model. 
The Y part of data is defined by having y=1 for meme and y=0 for notes.
We split the data into the three different parts in given percentage as mentioned above using 'train_test_split' function from 'sklearn' library.
Since this fuction split data in two parts, hence we have to use this function two times.
First time it splits 75% data into train and 25% into pre-test data.
Then this pre-test data is again splitted into two equal parts named dev/validation set and test set respectively of 12.5% total data.
These process is done over both meme and note dataset.

Such splitted data of images (X) and output (Y) are stacked together to form pre-shuffled data. Now we have stacked up the data, it is inthe form first 'k' images are of meme 
and other 'k' images are notes. Such data is not used as input to model since it is not randomly stacked, therfore causes difficulty in model learning.
Hence we randomly suffle the data which uses 'seed' function so that the state by which X is shuffled, Y is also shuffled in the same state.
These shuffled data is our dataset which can be used by model for learning and predicting.

I have defined many functions which are used in actual model. When we run the model with train data set, the model first passes through forward function which calculated a 
sample output. These output is then compared to actual output (Y) and the cost (loss) is calculated. Using cost we calculte gradient of each parameter.
These gradients are used to update the parameters. Since one time run is not sufficient to train the model and to minimise the cost, we run the model over train data for several 
times. These run are refered to 'epoch' or 'iterations'. Here also plot the graph between cost and epoch to understand the flow of training.

After the model is trained, we need to verify the model. We use validation set to verify the training and if required we also use this data to again update parameters.
Since model has been trained and verified, we can now use it to predict the type of image. 


### Convolutional Neural Network using Pytorch Library
In this method, each data image is conveted to torch tensor. The coloured images are not processed to form column arrays. 
This method uses image in the form of 3 channel data or 1 channel data as provided by user. In this method the splitting and proceesing over data is same.
In this method, I used predefined machine learning class and function from Pytorch Library eg. conv2d, MaxPool2d etc.
This method uses convolution to pass image features to another layer. Convolution includes processing a parts of image using a parameter matrix separately over the whole image 
matiex and channels. Therefore we are able to transfer features like vertical edges, horizontal edges etc to other layers. 
The advantage of this method over Numpy method is this method requires lesser number of parameters and it is more reliable and efficient. 

#### Other specifications
There is a term called 'Hyperparameters' which is used for parameters which are manually set and has to be tuned to enhance the model performance.
The following are the hyperparameters used in above two methods:
###### 1. Learning Rate used for parameter update
###### 2. Beta (beta1 and beta2) used in Adam optimisation
###### 3. Epsilon used in Adam Optimisation
###### 4. Lambda (refered as 'lambd' in programme) used in regularization of cost
According to the accuracy of model over training set, dev set adn test set, we need to tune this hyperparameters. 
The conditions in which hyperparameters are tuned are as follows:
###### 1. High Bias
This condition is known as 'underfitting'. High bias is defined as high training and dev set error or low accuracy. 
In this case, you need to use a bigger network with more layers and more neurons. You can also change Learning Rate to check for performance of model over them so that we can 
tune the model with learning rate which provides us more accuracy.
###### 2. High Variance
This condition is known as 'overfitting'. High variance is defined as high training set accuracy and low dev set accuracy.
In this case, you need more data to train model so that it learns more effectively. Even you can apply regularization to cost if not applied. 
If regularization is applied, then change the value of lambda to tune it.
###### 3. High Bias and High variance
This is a non ideal condition. This condition suggesting that model is not perfectly trained and more data is also required.

#### Precautions
While defining the model, number of neurons in the layers should gradually decrease. This helps model to properly transfer the features of image to next layers.
If not correctly defined, the accuracy of model decrease. This is one of the parameter which also has to be tuned. Increasing or decreasing number of layers, 
increasing or decreasing number of neurons in a given layer also helps to make model more efficient.

