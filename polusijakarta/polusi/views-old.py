from django.shortcuts import render
from sklearn import preprocessing
import pandas as pd
import numpy as np
import xlrd
import scipy.linalg as la
import matplotlib as mpl
import math as m
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine
import pymysql
from polusi import preproses as preproses


# Create your views here.

def home(request):
    return render(request, "home.html")


def training(request):
    result = []
    if request.method == 'POST':
        file = request.FILES['file']
        df = pd.read_excel(file)
        tanggal = df['tanggal']
        pm10 = df['pm10']
        so2 = df['so2']
        co = df['co']
        o3 = df['o3']
        no2 = df['no2']

        lokasi = df['lokasi']
        df.loc[df['lokasi'] == 'DKI1', 'lokasi2'] = 0
        df.loc[df['lokasi'] == 'DKI2', 'lokasi2'] = 1
        df.loc[df['lokasi'] == 'DKI3', 'lokasi2'] = 2
        df.loc[df['lokasi'] == 'DKI4', 'lokasi2'] = 3
        df.loc[df['lokasi'] == 'DKI5', 'lokasi2'] = 4
        lokasi2 = df['lokasi2']

        category = df['category']
        df.loc[df['category'] == 'Baik', 'category2'] = 0
        df.loc[df['category'] == 'Sedang', 'category2'] = 1
        df.loc[df['category'] == 'Tidak Sehat', 'category2'] = 2
        df.loc[df['category'] == 'Sangat Tidak Sehat', 'category2'] = 3
        df.loc[df['category'] == 'Bahaya', 'category2'] = 4
        category2 = df['category2']

        # dict = {'lokasi2': lokasi2, 'category2': category2, 'tanggal': tanggal, 'lokasi': lokasi, 'pm10': pm10,  'so2': so2,
        #         'co': co, 'o3': o3, 'no2': no2, 'category': category}
        # df = pd.DataFrame(dict)
        # engine = create_engine(
        #     'mysql+pymysql://root:@localhost/sys')
        # df.to_sql('data', con=engine,
        #           if_exists='replace', index=False)

        # connection = pymysql.connect(host='localhost',
        #                              user='root',
        #                              password='',
        #                              db='sys')

        # # create cursor
        # cursor = connection.cursor()

        # cursor = connection.cursor()
        # sql = "SELECT * FROM `data`"
        # cursor.execute(sql)

        # result = cursor.fetchall()

        dataset = df.values
        x = dataset[:, 4:9]
        y = dataset[:, 1]

        min_max_scaler = preprocessing.MinMaxScaler()

        X_scale = min_max_scaler.fit_transform(x)

        X_scale

        min_max_scaler.inverse_transform(X_scale)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scale, y, test_size=0.2)

        X_train

        y_train = y_train.astype(np.float32)

        y_test = y_test.astype(np.float32)

        xdata = preproses.ELM.random()  # df = pd.DataFrame()
        # print(xdata)
        # # add the array to df as a column
        # df = pd.DataFrame()
        # df['xdata'] = xdata
        # df['x_test'] = x_test
        # df['y_test'] = y_test
        # df['y_pred_svm'] = y_pred_svm

        # dict = {'xdata':xdata, 'x_test': df['x_test'], ' y_test': df['y_test'],'y_pred_svm':df['y_pred_svm']}

    return render(request, "training.html", {'result': xdata})
