from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
app=Flask(__name__)
modelpath='/Users/jasleengill/Desktop/cob_deploy/model.h5'
model =load_model(modelpath)

def model_predict(img_path, model):
    img =image.load_img(img_path, target_size=(100, 100)) 
    x= image.img_to_array(img)
    x= np.expand_dims(x,axis=0)
    x= tf.keras.applications.resnet50.preprocess_input(x) 
    preds= model.predict(x)
    val_class_names = ['potholes','graffitti','street cleanup','damaged signs','overflowing trashcan']
    predsText = val_class_names[np.argmax(preds)]
    return predsText
@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', file.filename)
        file.save(file_path)
        preds = model_predict(file_path, model)
        os.remove(file_path)
        return render_template('result.html', result=preds)
if __name__=='__main__':
    app.run(debug=True)