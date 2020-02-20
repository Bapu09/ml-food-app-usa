from flask import Flask, g, render_template, url_for, redirect, flash, make_response, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import model_from_json
import keras.backend.tensorflow_backend as tb
import spectral.io.envi as envi
import datetime 
from tensorflow.keras.optimizers import Adam
import os
import keras.backend as K
import numpy as np


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
        
        result = predictComponent(hdr_filename, img_filename, 'fat');
        print('Fat percentage : '+str(result))
        
        return render_template('success.html')

@app.route('/make_prediction', methods = ['GET'])
def make_prediction():
    hdr_filename = request.args.get('h')
    img_filename = request.args.get('i')
    component = request.args.get('c')
    res = predictComponent(hdr_filename, img_filename, component)
    return jsonify({'component_name' : component, 'component_value' : str(res)})
    
 
def predictComponent(hdr_filename, img_filename, componentName):
    #Checking Requested Model FIle Existance
    if(os.path.exists(os.path.join(app.config['MODEL_FOLDER'], componentName+'.json')) == False) :
        print("JSON File Doesn't Exists");
        
    if(os.path.exists(os.path.join(app.config['MODEL_FOLDER'], componentName+'.h5')) == False) :
        print("H5 File Exists");
        
        
    mask=np.load(os.path.join(app.config['MODEL_FOLDER'], componentName+'.npy'))
    m = load_model(os.path.join(app.config['MODEL_FOLDER'], componentName+'.json'), os.path.join(app.config['MODEL_FOLDER'], componentName+'.h5'))
    img = envi.open(os.path.join(app.config['UPLOAD_FOLDER'], hdr_filename), os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
    img = img[:,:,:]
    img1D = img_to_array(mask,img)
    mydate=datetime.datetime.now()

    opt = Adam(lr=0.001)
    m.compile(loss='mean_squared_error', optimizer=opt,metrics=[rmse,r2])
    result=m.predict(img1D)
    result = result[0,0]
    return result
    

def load_model(json,weights):
    json_file = open(json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights)#.h5 file
    return model


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
def r2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def img_to_array(mask,img,threshold=200):
    img = img[:, :, :]
    img_xy = np.mean(img, axis=2)
    img_th = np.where(img_xy > threshold, 1, 0)

    img_bs = img[img_th == 1]
    np_img = img_bs
    red_img = np.mean(np_img, 0)
    red_img = np.expand_dims(red_img,0)
    red_img=red_img[:,mask] 
    red_img = np.expand_dims(red_img,2)

    return red_img

