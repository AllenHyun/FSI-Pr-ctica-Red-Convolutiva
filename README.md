# FSI-Práctica-Red-Convolutiva

## Autora

María Cabrera Vérgez

## Tarea realizada

En esta práctica se va a desarrollar una red neuronal convolutiva para la clasificación de imágenes. Para ello, se debe utilizar el código de ejemplo inicial proporcionado en el material lectivo de la asignatura (redes_2.ipynb). Este código está escrito en PyTorch y será éste el framework que se utilice para esta práctica (no Keras ni Tensorflow).

A partir de este código inicial se deberán aplicar las transformaciones necesarias para adaptarlo a las condiciones del dataset elegido. Además, se requerirá realizar diferentes modelos modificando los hiperparámetros de la red (número de capas convolutivas y/o lineales, número de neuronas por capa, porcentaje de dropout, learninig rate, etc.) y estudiar y comparar los resultados obtenidos. El número de modelos con diferentes hiperparámetros dependerá del tiempo necesario para entrenarlos, como cifra orientativa podría ser un mínimo de cinco modelos.

## Lenguaje de Signos

El [dataset](https://drive.google.com/drive/folders/1KDZ62i4Z7o0-g36d1Lg6GWLD8jMlCbxW?usp=sharing) usado durante la realización de la práctica ha sido uno que contenía 5 clases diferentes, todas sobre lenguaje de signos (Yes, I Love You, No, Hello, Thank You). Los datos fueron sacados de la web kaggle (click [aquí](https://www.kaggle.com/datasets/mhmd1424/sign-language-detection-dataset-5-classes) para ver)

<img alt="kaggle_final" src="/images/kaggle_final.png">

El conjunto de datos también será dejado en un zip para su descarga, si se desea. Para poder probarlo, el proyecto debe de abrirse en Google Colab, se debe añadir a google drive el dataset y poner la ruta correcta. Se han realizado 5 experimentos: modelo MLP, modelo CNN con 5 épocas, modelo CNN con 20 épocas, modelo CNN con más capas y modificaciones, Transfer Learning. Las capturas de su funcionamiento se encuentran disponibles dentro del notebook.

## Conclusión

Cuantas menos imágenes se posee, más posible es que se presente un caso de overfitting. Esto es cuando el modelo empieza a memorizar las imágenes, más que aprender, porque ya no tiene de donde tirar. Con ciertos retoques, ese problema se puede ir reduciendo hasta casi hacerlo desaparecer del todo. Por otro lado, se nota que el entrenamiento del modelo usado para Transfer Learning es mayor, pues analiza las imágenes con facilidad y obtiene resultados muchos mejores. Los resultados ofrecidos en el primer experimento, con MLP son pobres, mostrando que es el peor modelo empleado en la práctica. CNN aumenta bastante los porcentajes y devuelve resultados mejores. 
