from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import time

model = load_model("agemodel.h5")
gender_dict = {0:'Male', 1:'Female'}


from flask import Flask ,render_template,request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html",predictions = None)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No files Uploaded'
    
    file = request.files['file']
    filename = file.filename
    imgpath = 'static/'+filename
    file.save(imgpath)
    time.sleep(2)
    print('image saved')
    img = tf.keras.utils.load_img(imgpath,color_mode = "grayscale")
    img = img.resize((128,128),Image.ANTIALIAS)
    img = np.array(img)
    img = img.reshape(1, 128, 128, 1)
    img = img/255
    pred = model.predict(img)
    val = pred[0][0][0]
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    predictions ={0:pred_age ,1:pred_gender,}
    if val<0.1:
        predictions ={0:"image not recognized" ,1:"image not recognized",}
    return render_template('home.html', predictions = predictions, image = imgpath)
app.run(debug=True)



