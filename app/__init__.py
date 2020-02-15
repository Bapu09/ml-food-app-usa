from flask import Flask, g, render_template, url_for, redirect, flash, make_response, request
from werkzeug.utils import secure_filename
from keras.models import model_from_json
import os

UPLOAD_FOLDER = './uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/upload', methods = ['POST']) 
def upload_image():
    if request.method == 'POST':  
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        
        #Now Let's load JSON
        m = load_model('./models/fat.json', './models/fat.h5')
        
        img = envi.open(os.path.join(app.config['UPLOAD_FOLDER'], filename),'media/uploads/rawfile1.img')
        img = img[:,:,:]
        img1D = img_to_array(img)
        mydate=datetime.now()
    
        opt = Adam(lr=0.001)
        m.compile(loss='mean_squared_error', optimizer=opt,metrics=[rmse,r2])
        result=m.predict(img1D)
        
        
        return render_template('success.html')
    
    
    
""" Independent Function """
def load_model(json,weights):
	json_file = open(json, 'r')#Json architecture file
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

def img_to_array(img,threshold=200):
    img = img[:, :, :]
    img_xy = np.mean(img, axis=2)
    img_th = np.where(img_xy > threshold, 1, 0)

    img_bs = img[img_th == 1]
    np_img = img_bs
    red_img = np.mean(np_img, 0)
    red_img = np.expands_dims(red_img,2)

    return red_img