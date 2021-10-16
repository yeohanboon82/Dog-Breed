import os
from app import app
from flask import flash, request, redirect, url_for, render_template
import numpy as np
import pandas as pd
import cv2              
import matplotlib.pyplot as plt                             
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.models import Sequential
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras import backend as K

# load list of dog names
dog_names = pd.read_csv('dog_names.csv')
dog_names = list(dog_names['breed'])

# define ResNet50 model
def gen_ResNet50_model():
    ResNet50_model = ResNet50(weights='imagenet')
    ResNet50_model._make_predict_function()
    return ResNet50_model

def extract_Resnet50(tensor):
    K.clear_session()
    result = ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
    K.clear_session()
    return result

def ResNet50_predict_labels(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    K.clear_session()
    ResNet50_model = gen_ResNet50_model()
    result = np.argmax(ResNet50_model.predict(img))
    K.clear_session()
    return result

#Load Dog Detector
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

#Load Face Detector
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

#Load the trained model.
def gen_ResNetTrained_model():
    ResNet_model = Sequential()
    ResNet_model.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
    ResNet_model.add(Dense(128, activation='relu'))
    ResNet_model.add(Dropout(0.5))
    ResNet_model.add(Dense(133, activation='softmax'))
    ResNet_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    ResNet_model.load_weights('saved_models/weights.best.ResNet50_final.hdf5')
    ResNet_model._make_predict_function()
    return ResNet_model

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    ResNet_model = gen_ResNetTrained_model()
    predicted_vector = ResNet_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    argmax_index = np.argmax(predicted_vector)
    confidence = predicted_vector[0][argmax_index]
    confidence = round(float(confidence),2)
    dog = dog_names[argmax_index]
    dog_class = dog.split('.')[0][-3::]
    dog_breed = dog.split('.')[1]
    dog_breed = dog_breed.replace('_',' ')
    K.clear_session()
    return confidence,dog_class,dog_breed

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
def identify_face(img):
    # convert image to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(grayscale_img, scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    face_box, face_coords = None, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(img,'Human Face',(x-50,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
        face_box = img[y:y+h, x:x+w]
        face_coords = [x,y,w,h]

    return img, face_box, face_coords

def predict_pictureV2(img_path,threshold):
    img = plt.imread(img_path)
    if dog_detector(img_path):
        confidence, dog_class, dog_breed = Resnet50_predict_breed(img_path)
        if confidence >= threshold:
            input_text = dog_breed +'('+str(confidence)+')'
            cv2.putText(img, input_text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3)
            message = 'This photo looks like a dog of breed ' + dog_breed +' .'

        else:
            message = 'This photo looks a dog but we do not know the breed.'

    elif face_detector(img_path):
        a, face_box, coords = identify_face(img)
        confidence, dog_class, dog_breed = Resnet50_predict_breed(img_path)
        input_text = dog_breed +'('+str(confidence)+')'
        cv2.putText(img, input_text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3)
        message = 'This photo looks like a human and resembles a ' + dog_breed +' .'

    else:
        message = 'Opps this photo looks like neither human nor dog.'
        
    plt.imshow(img)
    plt.savefig('static\predict\predict.jpg')
    return message

def predict_upload():
    img_path = 'static/uploads/upload.jpg'
    message = predict_pictureV2(img_path,0.7)
    return message

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():

    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = 'upload.jpg'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below. Press Predict.')
 
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/predict')
def predict():

    message = predict_upload()
    flash(message)

    return render_template(
        'predict.html'
    )



if __name__ == "__main__":
    app.run(debug=True)