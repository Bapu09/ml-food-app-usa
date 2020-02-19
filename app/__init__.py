from flask import Flask, g, render_template, url_for, redirect, flash, make_response, request
from werkzeug.utils import secure_filename
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import keras.backend.tensorflow_backend as tb

import os

UPLOAD_FOLDER = './uploads'
MODEL_FOLDER = './models'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/upload', methods = ['POST']) 
def upload_image():
    if request.method == 'POST':  
        hdrf = request.files['file_hdr']
        imgf = request.files['file_img']
        hdr_filename = secure_filename(hdrf.filename);
        img_filename = secure_filename(imgf.filename);
        
        hdrf.save(os.path.join(app.config['UPLOAD_FOLDER'], hdr_filename))
        imgf.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        
        
        #Now Let's load JSON
        print("Model Loading");
        
        if(os.path.exists(os.path.join(app.config['MODEL_FOLDER'], 'fat.json'))) :
            print("JSON File Exists");
        else:
            print("JSON File Doesn't Exists");
            
        if(os.path.exists(os.path.join(app.config['MODEL_FOLDER'], 'fat.h5'))) :
            print("H5 File Exists");
        else:
            print("H5 File Doesn't Exists");
            
            
        m = load_model(os.path.join(app.config['MODEL_FOLDER'], 'fat.json'), os.path.join(app.config['MODEL_FOLDER'], 'fat.h5'))
        print("Model Loaded");
        tb._SYMBOLIC_SCOPE.value = True
        img = envi.open(os.path.join(app.config['UPLOAD_FOLDER'], hdr_filename), os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        print("Image Loaded");
        img = img[:,:,:]
        img1D = img_to_array(img)
        print("Image to Array converted");
        mydate=datetime.now()
    
        opt = Adam(lr=0.001)
        m.compile(loss='mean_squared_error', optimizer=opt,metrics=[rmse,r2])
        print("Model Complied");
        result=m.predict(img1D)
        print("Model Predicted");
        print("Result "+result)
        
        return render_template('success.html')
    
    
    

def load_model(json,weights):
    print("Reading JSON File")
    json_file = open(json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    print("Reading JSON Complete")
    model = model_from_json(loaded_model_json)
    print("Loading Weight File")
    model.load_weights(weights)#.h5 file
    print("Weight File Loaded")
    return model


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
def r2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

