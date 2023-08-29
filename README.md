# Tumor Classification 🏥

<img src="img.jpg" width="800"/>

## Binary Malignant - Benign Tumor Classification"

The Model predicts if a Tumor image is Malignant or Benign

It was trained using Transfer Learning, with the VGG-16 Model. The last output layer was removed and then updated with a output layer of three neurons, with a `softmax` activation function. 

The Model was retrained with 1000 images from each class.

## Check-it out
Test the Model by yourself running the `main.py` file, built with `Streamlit`.

Remember first to run the model `jupyter notebook` file and save it.

<img src="preview.jpg" width="800"/>

## Run
```sh
streamlit run app.py
```


