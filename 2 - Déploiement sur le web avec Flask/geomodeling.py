#!/usr/bin/python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, session, request, redirect, flash,send_from_directory,send_file
import os
import pickle
import math
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename

model=pickle.load(open("model_clean_xgboost.sav", "rb"))
UPLOAD_FOLDER = '/home/bdiouf/Bureau/Mastere Spécialisé Big data/Formation MS big data 2020 Télécom Paris/Projet Bearing POint/Livrables école/Période 2/Déploiement sur le web/stockage_csv_web'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'012(\xffb\xb4\xd6_\xbe\x16\x10\xb9\x91\x1e\xf1'  #os.urandom(16,seed=2)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET','POST'])
def results():
    if request.method=="POST":
        req=request.form
        missing=[]
        for k,v in req.items():
            if v=="":
                missing.append(k)
        if 'GR' in missing:
            GR=np.nan
        else:
            GR=float(request.form["GR"])
        if 'CAL' in missing:
            CAL=np.nan
        else:
            CAL=float(request.form["CAL"])
        if 'RESD' in missing:
            RESD=np.nan
        else:
            RESD=math.log(float(request.form["RESD"]))
        if 'RESM' in missing:
            RESM=np.nan
        else:
            RESM=math.log(float(request.form["RESM"]))
        if 'PHIN' in missing:
            PHIN=np.nan
        else:
            PHIN=float(request.form["PHIN"])

        df = pd.DataFrame(data=[[GR, PHIN, RESD, RESM,CAL]],columns=["GR","PHIN","RESD","RESM","CAL"])
        col = ["GR","PHIN","RESD","RESM","CAL"]
        X_test = df[col]
        pred=model.predict(X_test)
        proba=model.predict_proba(X_test)
        if pred[0]==0 :
            flash("Avec une probablité de  {} la lithologie est du non argile".format(proba[0,0]))
            return render_template('index.html')
        else:
            flash("Avec une probablité de  {} la lithologie est de l'argile".format(proba[0,1]))
            return render_template('index.html')
    return redirect('index.html')

@app.route("/upload",methods=["GET","POST"])
def upload():
    if request.method=="POST":
        target=app.config['UPLOAD_FOLDER']
        print(target)

        file= request.files['file']
        print(file)
        filename = secure_filename(file.filename)
        destination=os.path.join(app.config['UPLOAD_FOLDER'],"test.csv")
        print(destination)
        file.save(destination)
        return render_template("complete.html")
    return render_template("upload.html")

@app.route("/return_files")
def download():
    data=pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],"test.csv"),sep=" ")
    data=data.replace(r'^\s*$', np.nan, regex=True)
    for i in [-9999,-9999.0,-999,-999.0] :
        data=data.replace(i,np.nan,regex=True)
    for i in ['RESD','RESM']:
        for j in range(len(data)):
            if data.loc[j,str(i)]==np.nan and data.loc[j,str(i)] > 0:
                data.loc[j,str(i)]=math.log(data.loc[j,str(i)])
            else:
                data.loc[j,str(i)]=np.nan
    x_test=data[["GR","PHIN","RESD","RESM","CAL"]]
    pred=model.predict(x_test)
    proba=model.predict_proba(x_test) 
    proba_sable=pd.DataFrame(proba[:,0],columns=['probablité sable'])
    proba_argile=pd.DataFrame(proba[:,1],columns=['probablité argile'])
    litho_predite=pd.DataFrame(pred,columns=['lithologie prédite'])
    #dict_={"0":"Sable","1":"Argile"}
    #for i in [0,1]:
    #    litho_predite.replace(i,dict_[str(i)])
    result_test=pd.concat([x_test,proba_sable,proba_argile,litho_predite],axis=1)   
    result_test.to_csv(os.path.join(app.config['UPLOAD_FOLDER'],"result_test.csv"))  
    return send_file('stockage_csv_web/result_test.csv',as_attachment=True) #send_from_directory(directory='stockage_csv_web', filename='result_test.csv"', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)



