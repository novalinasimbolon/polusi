from django.shortcuts import render
from sklearn import preprocessing
import pandas as pd
import numpy as np
# import xlrd
import scipy.linalg as la
import base64
import io
from matplotlib.figure import Figure
import matplotlib as mpl
import matplotlib.pyplot as plt
import math as m
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine
import pymysql
from polusi import preproses as preproses
from sklearn import preprocessing
from sklearn import metrics
import string
from sklearn.metrics import confusion_matrix
import seaborn as sn


# Create your views here.

def home(request):
    return render(request, "home.html")


def classify(x):
    # x.loc[x['lokasi'] == 'DKI1', 'lokasi'] = 0
    # x.loc[x['lokasi'] == 'DKI2', 'lokasi'] = 1
    # x.loc[x['lokasi'] == 'DKI3', 'lokasi'] = 2
    # x.loc[x['lokasi'] == 'DKI4', 'lokasi'] = 3
    # x.loc[x['lokasi'] == 'DKI5', 'lokasi'] = 4
    x.loc[x['category'] == 'Baik', 'category'] = 0
    x.loc[x['category'] == 'Sedang', 'category'] = 1
    x.loc[x['category'] == 'Tidak Sehat', 'category'] = 2
    x.loc[x['category'] == 'Sangat Tidak Sehat', 'category'] = 3
    x.loc[x['category'] == 'Bahaya', 'category'] = 4


def training(request):
    X_train = []
    if request.method == 'POST':
        file = request.FILES['file']
        df = pd.read_excel(file)
        classify(df)

        df[["tanggal", "pm10", "so2", "co", "o3", "no2",
            "lokasi", "category"]]

        dataset = df.values
        X = dataset[:]
        y = dataset[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=29)

    return render(request, "training.html", {'result': X_train})


def testing(request):
    akurasi = []
    # X_testText = []
    # X_test2 = []
    # y_test2 = []
    # y_pred2 = []
    # semua_data = []
    Xtest = []
    ytest = []
    ypred = []
    result = []
    result_cm = []
    response = []
    if request.method == 'POST':
        file = request.FILES['file']
        df = pd.read_excel(file)
        classify(df)

        df[["tanggal", "pm10", "so2", "co", "o3", "no2",
            "lokasi", "category"]]

        dataset = df.values

        Xtext = dataset[:]
        X = dataset[:, 1:6]
        y = dataset[:, -1]

        X_trainText, X_testText, y_trainText, y_testText = train_test_split(
            Xtext, y, test_size=0.2, random_state=29)

        min_max_scaler = preprocessing.MinMaxScaler()
        X_scale = min_max_scaler.fit_transform(X)
        data = X_scale

        # Create instance of ELM object with 100 hidden neuron
        elm = preproses.ELM(data.shape[1], 1, 100)
        # Train test split 80:20
        X_train, X_test, y_train, y_test = train_test_split(
            data, y, test_size=0.2, random_state=29)

        # Train data
        elm.train(X_train, y_train.reshape(-1, 1))

        # Make prediction from training process
        y_pred = elm.predict(X_test)
        y_test = y_test.astype(int)
        y_pred = y_pred.astype(int)

        for i in range(len(y_pred)):
            # semua_data.append(
            #     [X_testText[i], y_test[i], y_pred[i]])
            Xtest.append(X_testText[i])
            df2 = pd.DataFrame(Xtest)
            tanggal = df2[0]
            pm10 = df2[1]
            so2 = df2[2]
            co = df2[3]
            o3 = df2[4]
            no2 = df2[5]
            lokasi = df2[6]
            category = df2[7]

            ytest.append(y_test[i])
            ypred.append(y_pred[i])

            list2 = []
            for i in range(len(ypred)):
                t = int(ypred[i])
                list2.append(t)

            # xtest = df['X_testText']
            # ytest = df['y_test']
            # ypred = df['y_pred']
            dict = {'tanggal': tanggal, 'pm10': pm10, 'so2': so2,
                    'co': co, 'o3': o3, 'no2': no2, 'lokasi': lokasi,
                    'category': category, 'ytest': ytest, 'ypred': list2}
            df = pd.DataFrame(dict)
            # df = pd.DataFrame(semua_data)

            engine = create_engine(
                'mysql+pymysql://root:@localhost/sys')
        df.to_sql('testing_data_result', con=engine,
                  if_exists='replace', index=False)

        connection = pymysql.connect(host='localhost',
                                     user='root',
                                     password='',
                                     db='sys')

        cursor = connection.cursor()
        sql = "SELECT * FROM `testing_data_result`"
        cursor.execute(sql)

        result = cursor.fetchall()

        # X_test2.append(X_testText[i])
        # y_test2.append(y_test[i])
        # y_pred2.append(y_pred[i])
        # semua_data.append([X_test2, y_test2, y_pred2])

        print('Accuracy: ', accuracy_score(y_test, y_pred))
        report_dict = metrics.classification_report(
            y_test, y_pred, output_dict=True)
        print(report_dict)
        akurasi.append(accuracy_score(y_test, y_pred))

        report_dict = metrics.classification_report(
            y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report_dict)
        engine = create_engine(
            'mysql+pymysql://root:@localhost/sys')
        df.to_sql('confusion_matrix', con=engine,
                  if_exists='replace', index=False)

        result_cm = preproses.confusion_matrix()

        labels = ["STS",
                  "TS", "Sedang", "Baik"]
        cf_matrix = confusion_matrix(y_test, y_pred)
        ax = sn.heatmap(cf_matrix, annot=True, xticklabels=labels,
                        yticklabels=labels, cmap="YlGnBu", fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Actual')
        fig = Figure(figsize=(8, 6))
        plt.axis('image')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        response = base64.b64encode(buf.getvalue()).decode(
            'utf-8').replace('\n', '')
        buf.close()

    return render(request, "testing.html", {'result': result, 'response': response, 'result_cm': result_cm})
