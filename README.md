# FSI-Práctica-Red-Convolutiva

## Autora

María Cabrera Vérgez

## Tarea realizada

En esta práctica se va a desarrollar una red neuronal convolutiva para la clasificación de imágenes. Para ello, se debe utilizar el código de ejemplo inicial proporcionado en el material lectivo de la asignatura (redes_2.ipynb). Este código está escrito en PyTorch y será éste el framework que se utilice para esta práctica (no Keras ni Tensorflow).

A partir de este código inicial se deberán aplicar las transformaciones necesarias para adaptarlo a las condiciones del dataset elegido. Además, se requerirá realizar diferentes modelos modificando los hiperparámetros de la red (número de capas convolutivas y/o lineales, número de neuronas por capa, porcentaje de dropout, learninig rate, etc.) y estudiar y comparar los resultados obtenidos. El número de modelos con diferentes hiperparámetros dependerá del tiempo necesario para entrenarlos, como cifra orientativa podría ser un mínimo de cinco modelos.

## Lenguaje de Signos

El [dataset](https://drive.google.com/drive/folders/1KDZ62i4Z7o0-g36d1Lg6GWLD8jMlCbxW?usp=sharing) usado durante la realización de la práctica ha sido uno que contenía 5 clases diferentes, todas sobre lenguaje de signos (Yes, I Love You, No, Hello, Thank You). Los datos fueron sacados de la web kaggle (click [aquí](https://www.kaggle.com/datasets/mhmd1424/sign-language-detection-dataset-5-classes) para ver).

<img alt="kaggle_final" src="/images/kaggle_final.png">

El conjunto de datos no puede ser colgado en github debido a su tamaño, pero puede ser descargado desde el link anteriormente dado. Para poder probarlo, el proyecto debe de abrirse en Google Colab, se debe añadir a google drive el dataset y poner la ruta correcta. Se han realizado 5 experimentos: modelo MLP, modelo CNN con 5 épocas, modelo CNN con 20 épocas, modelo CNN con más capas y modificaciones, Transfer Learning. Las capturas de su funcionamiento se encuentran disponibles dentro del notebook. El archivo editado en Google Colab es [este](https://colab.research.google.com/drive/1K15u3zp4uFYMX3gFHsi8e7O75yLJAAlL?usp=sharing)

## Conclusión

Cuantas menos imágenes se posee, más posible es que se presente un caso de overfitting. Esto es cuando el modelo empieza a memorizar las imágenes, más que aprender, porque ya no tiene de donde tirar. Con ciertos retoques, ese problema se puede ir reduciendo hasta casi hacerlo desaparecer del todo. Por otro lado, se nota que el entrenamiento del modelo usado para Transfer Learning es mayor, pues analiza las imágenes con facilidad y obtiene resultados muchos mejores. Los resultados ofrecidos en el primer experimento, con MLP son pobres, mostrando que es el peor modelo empleado en la práctica. CNN aumenta bastante los porcentajes y devuelve resultados mejores. 

## Webgrafía

WEBGRAFÍA FSI

https://docs.python.org/3/library/os.path.html
https://pypi.org/project/albumentations
https://www.w3schools.com/python/ref_os_walk.asp
https://stackoverflow.com/questions/79146746/clarifying-batch-size-when-using-multiple-gpus#:~:text=batch_size%20in%20the%20DataLoader%20specifies,Each%20GPU%20processes%20512%20samples
https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
https://docs.pytorch.org/docs/stable/data.html
https://phuijse.github.io/MachineLearningBook/contents/neural_networks/transfer_learning.html
https://interactivechaos.com/es/manual/tutorial-de-machine-learning/one-hot-encoding
https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tight_layout.html
https://www.reddit.com/r/learnmachinelearning/comments/9ukdvc/what_error_might_happen_when_my_learning_rate_is/
https://gamco.es/glosario/percepatron-multicapa-mlp/
https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
https://www.ultralytics.com/es/glossary/relu-rectified-linear-unit#:~:text=La%20no%20linealidad%20introducida%20por,datos%20médicos%20de%20alta%20resolución.
https://stackoverflow.com/questions/50792316/what-does-1-of-view-mean-in-pytorch
https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
https://www.datacamp.com/es/tutorial/adam-optimizer-tutorial
https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
https://docs.pytorch.org/docs/stable/generated/torch.flatten.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
https://keepcoding.io/blog/batch-normalization-red-convolucional/
https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
https://github.com/Leslie-Romero/VC_P3_LRM_MCV/blob/main/Tareas.ipynb
