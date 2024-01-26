# Reconocimiento_human_activity
<img src="./rsc/67440cf78c8aedfa65f106d20d1cbdd8.gif" alt="GIF" width="100%">


Welcome to the project on recognition of human activity. This project consists of creating an AI model to predict the activity being performed at each moment.

<div>
    This will be achieved by utilizing data from the sensors of smartphones, the same data and features detailed in the data collection. The project's aim is not to develop the app itself but to create the AI responsible for the task. However, I will also develop an API that can be used in a cloud-connected app.
</div>

## Data collect:

##### The data collect was carried out in the following way; having a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data. 
##### The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.

Check the README.txt file for further details about this dataset. 
<div>A video of the experiment including an example of the 6 recorded activities with one of the participants can be seen in the following link: http://www.youtube.com/watch?v=XOEN9W05_4A</div>

> Credits:<br>Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra y Jorge L. Reyes-Ortiz. Un Conjunto de Datos de Dominio Público para el Reconocimiento de Actividades Humanas Utilizando Teléfonos Inteligentes. 21º Simposio Europeo sobre Redes Neuronales Artificiales, inteligencia Computacional y Aprendizaje Automático, ESANN 2013. Brujas, Bélgica, del 24 al 26 de abril de 2013.

## Extractrion, Transformation and Load(ETL):

In the data documentation i found that the dataset in theory consists of 561 features but when i load the data i found one single feature, because the rows are taken as complete strings, not as separate columns. So I started by making the following code to get all the columns or features:

```
# Transformations of Xtrain
numeros = []
for sublist in Xtrain:

    numeros_sublist = []

    for string_num in sublist:

        for substring in string_num.split(" "):

          for word in substring.split():

            numeros_sublist.append(float(word))

    numeros.append(numeros_sublist)

array_resultante = np.array(numeros)

print(array_resultante)
```

What this code does is take each row as a string, and using the 'split()' method, it extracts each word in the row, transforms it into a float, and then saves the coordinates in a list using the ['append()'](https://docs.python.org/3/tutorial/datastructures.html) method. Finally, each list is stored in a list of lists, and the entire structure is converted into an [array](https://numpy.org/doc/stable/reference/generated/numpy.array.html) with the shape shown bellow. 

```
shape: (7351, 561)

[[ 0.27841883 -0.01641057 -0.12352019 ... -0.8447876   0.18028889
  -0.05431672]
 [ 0.27965306 -0.01946716 -0.11346169 ... -0.84893347  0.18063731
  -0.04911782]
 [ 0.27917394 -0.02620065 -0.12328257 ... -0.84864938  0.18193476
  -0.04766318]
 ...
 [ 0.27338737 -0.01701062 -0.04502183 ... -0.77913261  0.24914484
   0.04081119]
 [ 0.28965416 -0.01884304 -0.15828059 ... -0.78518142  0.24643223
   0.02533948]
 [ 0.35150347 -0.01242312 -0.20386717 ... -0.78326693  0.24680852
   0.03669484]]
```
To the xtest dataset i do the same, the results are shown bellow:

```
shape: (2946, 561)

[[ 0.28602671 -0.01316336 -0.11908252 ... -0.69809082  0.28134292
  -0.08389801]
 [ 0.27548482 -0.02605042 -0.11815167 ... -0.70277146  0.28008303
  -0.0793462 ]
 [ 0.27029822 -0.03261387 -0.11752018 ... -0.69895383  0.28411379
  -0.077108  ]
 ...
 [ 0.34996609  0.03007744 -0.11578796 ... -0.65535684  0.27447878
   0.18118355]
 [ 0.23759383  0.01846687 -0.09649893 ... -0.65971859  0.26478161
   0.18756291]
 [ 0.15362719 -0.01843651 -0.13701846 ... -0.66008023  0.26393619
   0.1881034 ]]
```
For all transformations, I use Numpy and Python utilities, except for the 'one-hot encoding' of the y_train and y_test, where I use TensorFlow to perform the 'one-hot encoding', concretly the ["to_categorical()"](https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical) function from keras.untils. The code shown bellow:

```
#We applied "one hot encoding" to categorical classes
Ytrain_one_hot = to_categorical(Ytrain - 1, num_classes=6)
Ytest_one_hot = to_categorical(Ytest - 1, num_classes=6)
```

> NOTE: All transformations can be seen in the notebook [exploraction.ipynb](./exploraction.ipynb), except for the one-hot encoding, which you can find in the [IA_development.ipynb](./IA_development.ipynb) notebook.


## AI development

For the model, I chose a neural network because neural networks can be trained with various types of data. In this case, the data consist of signals from the sensors of a smartphone, making it the best option for the task. I started by applying a PCA algorithm, and the next step was to generate different neural network (NN) architectures to choose the best one. I explain with more details below:

<div>
  I decided to compress the data using the PCA algorithm to visualice the data.
</div>


You can find more details in the notebook [IA_development.ipynb](./IA_development.ipynb) or in the IA_details.md file. The model is available in this repository or by clicking on [model](./trained_model).

