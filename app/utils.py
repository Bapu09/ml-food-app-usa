
from keras.models import model_from_json
import spectral as sp 
import spectral.io.envi as envi
import numpy as np 

def load_model(json,weights):
	json_file = open(json, 'r')#Json architecture file
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights(weights)#.h5 file
	return model

def handle_uploaded_file(f,path):
    with open(path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def upload(request):
	hdr = ['hdr']
	raw = ['raw','img']
	data={}
	if(request.method=='GET'):
		return render(request,'home/welcome.html',data)
	else:
		files = request.FILES
		h = files.getlist('file')[0]
		c = files.getlist('file')[1]
		
		if str(h)[-3:] in raw:
			t=h
			h=c
			c=t
		
		print(h,c)
		handle_uploaded_file(h,'media/uploads/hdrfile1.hdr')
		handle_uploaded_file(c,'media/uploads/rawfile1.img')
		print("Done")
		img = envi.open('media/uploads/hdrfile1.hdr','media/uploads/rawfile1.img')
		b = np.random.randint(img.metadata.get('bands'),size=3)
		if ('default bands' in img.metadata.keys()):
			b = img.metadata.get('default bands')
		iR=int(b[0])
		iG=int(b[1])
		iB=int(b[2])


		uid = str(uuid.uuid1())

		sp.save_rgb('home/static/image_{}.jpg'.format(uid),img,[iR,iG,iB])
		request.session['upload']=1
		request.session['uid']=uid
		request.session['Bands'] = img.metadata.get('bands')
		request.session['Samples'] = img.metadata.get('samples')
		request.session['lines'] = img.metadata.get('lines')

		return HttpResponseRedirect(reverse('home'))

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

def analysis(request):

	m = settings.MODEL
	
	img = envi.open('media/uploads/hdrfile1.hdr','media/uploads/rawfile1.img')
	img = img[:,:,:]
	img1D = img_to_array(img)
	mydate=datetime.now()

	opt = Adam(lr=0.001)
	m.compile(loss='mean_squared_error', optimizer=opt,metrics=[rmse,r2])
	result=m.predict(img1D)

	return result