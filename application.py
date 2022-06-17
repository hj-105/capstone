from flask import Flask, flash, g, redirect, render_template, request, url_for, send_from_directory, sessions
from werkzeug.utils import secure_filename
import pyscreenshot as ImageGrab
from PIL import Image
import shutil
import flask
import time
import sys
import os

# DL
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
# from tensorflow import keras
#import matplotlib.pyplot as plt

#DB
# pymysql.install_as_MySQLdb()
# import dbModule
import pymysql
from flask import current_app as current_app
from flask import Blueprint, request, render_template, flash, redirect, url_for

#PATH
input_img_dir='./uploads/'
model_dir='./model/_105-0.6621.hdf5'
hair_removed_dir='./static/output/'

#IMG SIZE
IMG_SIZE=(224, 224)

#PREPROCESSING FUNCTION
def hair_remove(filename):
    #fname = [s for s in (os.listdir(img_dir)) if s.endswith('.jpg') or s.endswith('.png')]
    img = cv2.imread(input_img_dir+filename)
    ## 전처리 1. hair remove     
    # 이미지 컬러를 그레이스케일로 변환하기 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1,(17,17))

    # Perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting 
    ret,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    final_image = cv2.inpaint(img,threshold,1,cv2.INPAINT_TELEA)

    # save hair remove img
    cv2.imwrite(hair_removed_dir+filename,final_image)
    print('=======')
    print('hair remove save done'+hair_removed_dir+filename)

    return hair_removed_dir+filename

def model(filename,model_weight_dir):
    
    # load model 
    print("---"*30)
    print('load model...') 
        
    model = load_model(model_weight_dir)
        
    print('load model done') 
    # 1.이미지 로드
    image_path = hair_remove(filename)
        
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    # 2.이미지 어레이로 변환
    img = tf.keras.preprocessing.image.img_to_array(img)
    # 3. 이미지 정규화
    img = img / 255
    # 4.이미지 배치 확장
    img_array = tf.expand_dims(img, 0)

    predict =model.predict(img_array)[0]
    predict=predict*100

    melanoma=predict[0]
    nv=predict[1]
    bkl=predict[2]

    if np.max(predict)==(melanoma):
        finding = "Diagnosis: MELANOMA: {:.2f}%".format(melanoma)
    elif np.max(predict)==(nv):
        finding = "Diagnosis: NV: {:.2f}%".format(nv)
    elif np.max(predict)==(bkl):
        finding = "Diagnosis: BKL: {:.2f}%".format(bkl)

    print('=======')
    print('model prediction is done')

    return finding,melanoma,nv,bkl

#DB
global db, cursor

db = pymysql.connect(host='localhost', port=3306, user='root', db='22_db', password='1234', charset='utf8')
cursor = db.cursor(pymysql.cursors.DictCursor)   
 
# def execute( query, args={}):
#     cursor.execute(query, args)   
     
# def executeOne( query, args={}):
#     cursor.execute(query, args)
#     row=cursor.fetchone()
#     return row

# def executeAll( query, args={}):
#     cursor.execute(query, args)
#     row=cursor.fetchall()
#     return row

# def commit():
#     db.commit()

#FLASK
application = Flask(__name__)

#link to html
@application.route("/")
def login():
    return render_template("login.html")

@application.route("/log", methods=['GET', 'POST'])
def log():
    if request.method == 'POST':
        name = request.form['name']
        id = request.form['id']
        birth = int(request.form['birth'])
        
        # mysql    
        sql="INSERT INTO 22_db.info(생년월일,이름,ID) values('%d','%s','%s');"%(birth,name,id)
        cursor.execute(sql)
        db.commit()
        
        sql="SELECT * FROM INFO"
        
        return render_template("hello.html",name=name,id=id,birth=birth
                               ,result='insert done',resultData=None, resultUPDATE=None)
    else:
        return render_template("login.html",name=None,id=None,birth=None)

@application.route("/apply")
def apply():
    return render_template("apply.html")

@application.route("/app", methods=['GET', 'POST'])
def app():
    if request.method == 'POST':
        sex = request.form['sex']
        itch = request.form['itch']
        pain = request.form['pain']
        area = request.form['area']
        
        # mysql에 데이터 넣을 코드
        sql="UPDATE 22_db.info SET 성별= '%s', 가려움='%s', 통증 = '%s', 위치= '%s', 날짜=now() ORDER BY idxx DESC LIMIT 1;"%(sex,itch,pain,area)
        cursor.execute(sql)
        db.commit()
        sql="SELECT * from INFO"
        row=cursor.fetchall()
        
        sql="SELECT * FROM INFO"
        
        return render_template("applyPhoto.html",sex=sex,itch=itch,pain=pain,area=area)
                            #    ,result=None, resultData=None, resutlUPDATE=row[0])
    else:
        return render_template("apply.html",sex=None,itch=None,pain=None,area=None)

    
# @application.route("/hello")
# def hello():
#     return render_template("hello.html")
   
@application.route("/us")
def us():
    return render_template("us.html")

@application.route("/skin")
def skin():
    return render_template("skin.html")

@application.route("/agree")
def agree():
    return render_template("agree.html")

@application.route("/applyPhoto")
def applyPhoto():
    return render_template("applyPhoto.html")

@application.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        f = request.files['file']
        f.save("./uploads/" + secure_filename(f.filename))

        src = os.path.join("./uploads/" + f.filename)
        dst = os.path.join("./static/input/" + f.filename)
        shutil.copyfile(src,dst)

        finding,melanoma,nv,bkl = model(f.filename, model_dir)
        # finding = "Prediction) NV: 95.80%"
        # melanoma,nv,bkl = 3.1762884,95.79547,1.0282432
        
        sql="UPDATE 22_db.info SET 사진= '%s', 결과= '%s' ORDER BY idxx DESC LIMIT 1;"%(dst,str(finding))
        cursor.execute(sql)
        db.commit()
        sql="SELECT * from INFO"
        row=cursor.fetchall()
        
        sql="SELECT * FROM INFO"

        return render_template('result.html', image_file = f.filename, finding=finding,mel=melanoma,nv=nv,bkl=bkl)
    else:
        return render_template('applyPhoto.html')

@application.route("/download")
def download():
    global time
    now = time.localtime()
    time = "%04d-%02d-%02d-%02dh-%02dm-%02ds" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    img=ImageGrab.grab()
    saveas="{}{}".format(time,'.png')
    img.save(saveas)
    return flask.send_file(saveas, as_attachment=True)


if __name__ == "__main__":
    application.debug = True
    application.run()

# if __name__ == "__main__":
#     application.debug = True
#     application.run(host='0.0.0.0')