import shutil

from flask import Flask, render_template, request, url_for, flash
import os
import pandas as pd
import numpy as np
from mysql.connector import cursor
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

from sklearn.datasets._base import load_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import mysql.connector
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import metrics
from werkzeug.utils import secure_filename

mydb = mysql.connector.connect(host='localhost',user="root",password="",port=3306,database="covid")


app=Flask(__name__)
app.config['UPLOAD_FOLDER']=r"uploads"
app.config['SECRET_KEY'] = 'the random string'
@app.route('/')
def index():

    return render_template('index.html', msg='success')

@app.route("/load_data",methods=['GET','POST'])
def load_data():
    print('------post--------')
    if request.method == "POST":
         myfile= request.files['file']
         ext = os.path.splitext(myfile.filename)[1]
         if ext.lower() == ".csv":
             shutil.rmtree(app.config['UPLOAD_FOLDER'])
             os.mkdir(app.config['UPLOAD_FOLDER'])
             myfile.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(myfile.filename)))
             flash('The data is loaded successfully', 'success')
             return render_template('load_data.html')
         else:
             flash('Please upload a CSV type document only', 'warning')
             return render_template('load_data.html', msg ='error')

    return render_template('load_data.html')

@app.route("/view_data",methods = ['GET','POST'])
def view_data():
    print("--------------services__data-----------")
    print('B')
    myfile = os.listdir(app.config['UPLOAD_FOLDER'])
    print('vv')
    global full_data
    print('jjjjjjjj')
    full_data=pd.read_csv(os.path.join(app.config["UPLOAD_FOLDER"], myfile[0]))
    temp=pd.DataFrame(full_data).sample(1000)


    return render_template('view_data.html', col=temp.columns.values, row_value = temp.values.tolist())
global d
global X
global y

@app.route('/model_cases',methods = ['GET','POST'])
def model_cases():
    if request.method == "POST":

        model_no =int(request.form['algo'])
        global testsize
        testsize=int(request.form['test_size'])
        testsize=testsize/100
        d = pd.read_csv("uploads/covid_main.csv")
        # print(d.head(2))
        # d=d[0:10000]
        X = d.drop(['Total_Confirmed_Cases'],axis =1)
        y = d[['Total_Confirmed_Cases']]

        # print(X_train.shape)
        # print(X_test.shape)
        # print(y_train.shape)
        # print(y_test.shape)



        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # scaler.fit_transform(X_train)
        # scaler.fit_transform(X_test)
        # scaler.fit_transform(y_train)
        # scaler.fit_transform(y_test)

        # X_train.head(10)

        print('Split_done')
        if model_no == 1:
            global X_train,X_test,y_train,y_test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=52)
            model=LinearRegression()
            model.fit(X_train,y_train)
            print('akjskdses')
            predictions = model.predict(X_test)
            # print(metrics.mean_absolute_error(y_test,predictions))
            # print(metrics.mean_squared_error(y_test,predictions))
            # np.sqrt(metrics.mean_squared_error(y_test,predictions))
            b = r2_score(y_test, predictions)
            print(b)
            return render_template('model_cases.html',msg="Error",b = b)
        elif model_no ==2:

            from numpy import arange
            # global prediction
            from sklearn.model_selection import RepeatedKFold
            from sklearn.model_selection import GridSearchCV
            # X_train, X_test, y_train, y_test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=52)
            model = Lasso(alpha=1.0)
            # cv = RepeatedKFold(n_splits=10,n_repeats=3,random_state=10)
            # grid = dict()
            # grid['alpha'] = arange(0,1,0.01)
            # search = GridSearchCV(model,grid,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
            # model = Lasso(alphas=arange(0,1,0.01),cv=cv,n_jobs=-1)
            model.fit(X_train,y_train)
            prediction = model.predict(X_test)
            # print(metrics.mean_absolute_error(y_test, prediction))
            # print(metrics.mean_squared_error(y_test, prediction))
            # np.sqrt(metrics.mean_squared_error(y_test, prediction))
            b = r2_score(y_test, prediction)

            return render_template('model_cases.html',msg="Error",b=b)
        elif model_no==3:

            from sklearn.svm import SVR
            global prd
            model=SVR(kernel='poly')
            print('bhurrrrrrrrrrrrrrrrrrr')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=52)
            p = pd.Series(y_train['Total_Confirmed_Cases'])
            q = pd.Series(y_test['Total_Confirmed_Cases'])
            model.fit(X_train[:2000], p[:2000])
            prd = model.predict(X_test[:2000])
            # print(mean_absolute_error(q, prd))
            # print(mean_squared_error(q, prd))
            #pd.np.sqrt(mean_squared_error(q, prd))
            b = r2_score(q[:2000],prd)
            return render_template('model_cases.html',msg ="Error",b=b)




    return render_template('model_cases.html')

@app.route('/Model_Deaths',methods = ['GET','POST'])
def Model_Deaths():
    print('NN')
    if request.method == "POST":
        model_no = int(request.form['algo'])
        testsize = int(request.form['test_size'])
        testsize = testsize / 100
        d = pd.read_csv("uploads/covid_main.csv")
        print(d.head(2))
        # d=d[0:10000]
        X = d.drop(['Total_Fatalities'], axis=1)
        y = d[['Total_Fatalities']]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=52)
        if model_no == 1:
            global X_train, X_test, y_train, y_test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=52)
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            # model=LinearRegression()
            # model.fit(X_train,y_train)
            # predictions = model.predict(X_test)
            print(metrics.mean_absolute_error(y_test,predictions))
            print(metrics.mean_squared_error(y_test,predictions))
            np.sqrt(metrics.mean_squared_error(y_test,predictions))
            b = r2_score(y_test, predictions)
            return render_template('Model_Deaths.html',message="Error",b = b)
        elif model_no ==2:

            from numpy import arange
            from sklearn.model_selection import RepeatedKFold
            from sklearn.model_selection import GridSearchCV

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=52)
            model = Lasso(alpha=1.0)
            # cv = RepeatedKFold(n_splits=10,n_repeats=3,random_state=10)
            # grid = dict()
            # grid['alpha'] = arange(0,1,0.01)
            # search = GridSearchCV(model,grid,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
            # model = Lasso(alphas=arange(0,1,0.01),cv=cv,n_jobs=-1)
            model.fit(X_train,y_train)
            prediction = model.predict(X_test)
            print(metrics.mean_absolute_error(y_test, prediction))
            print(metrics.mean_squared_error(y_test, prediction))
            np.sqrt(metrics.mean_squared_error(y_test, prediction))
            b = r2_score(y_test, prediction)

            return render_template('Model_Deaths.html',message="Error",b=b)
        elif model_no==3:


            model=SVR(kernel='poly')
            print('bhurrrrrrrrrrrrrrrrrrr')
            # print(y_train)
            # new_y=y_train['Total_Confirmed_Cases'].values()
            # model.fit(X_train,y_train)
            # prd=model.predict(X_test)
            # TEST_SIZE = 30
            p = pd.Series(y_train['Total_Fatalities'])
            q = pd.Series(y_test['Total_Fatalities'])
            model.fit(X_train[:1200], p[:1200])
            prd = model.predict(X_test[:1200])
            # print(mean_absolute_error(q, prd))
            # print(mean_squared_error(q, prd))
            #pd.np.sqrt(mean_squared_error(q, prd))
            b = r2_score(q[:1200],prd)
        return render_template('Model_Deaths.html', message ="Error",b=b)



    return render_template('Model_Deaths.html')



@app.route('/Model_Recoveries',methods = ['GET','POST'])
def Model_Recoveries():
    if request.method=="POST":
        model_no = int(request.form['algo'])
        testsize = int(request.form['test_size'])
        testsize = testsize / 100
        d = pd.read_csv("uploads/covid_main.csv")
        print(d.head(2))
        # d=d[0:10000]
        X = d.drop(['Total_Recovered_Cases'], axis=1)
        y = d[['Total_Recovered_Cases']]
        global X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=52)
        if model_no == 1:

            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            print(metrics.mean_absolute_error(y_test, predictions))
            print(metrics.mean_squared_error(y_test, predictions))
            np.sqrt(metrics.mean_squared_error(y_test, predictions))
            b = r2_score(y_test, predictions)
            return render_template('Model_Recoveries.html', msg="Error", b=b)
        elif model_no == 2:

            from numpy import arange
            from sklearn.model_selection import RepeatedKFold
            from sklearn.model_selection import GridSearchCV

            model = Lasso(alpha=0.01)
            # cv = RepeatedKFold(n_splits=10,n_repeats=3,random_state=10)
            # grid = dict()
            # grid['alpha'] = arange(0,1,0.01)
            # search = GridSearchCV(model,grid,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
            # model = Lasso(alphas=arange(0,1,0.01),cv=cv,n_jobs=-1)
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            print(metrics.mean_absolute_error(y_test, prediction))
            print(metrics.mean_squared_error(y_test, prediction))
            np.sqrt(metrics.mean_squared_error(y_test, prediction))
            b = r2_score(y_test, prediction)

            return render_template('Model_Recoveries.html', msg="Error", b=b)
        elif model_no == 3:

            from sklearn.svm import SVR
            model = SVR(kernel='poly')
            print('bhurrrrrrrrrrrrrrrrrrr')
            # print(y_train)
            # new_y=y_train['Total_Confirmed_Cases'].values()
            # model.fit(X_train,y_train)
            # prd=model.predict(X_test)
            # TEST_SIZE = 30
            p = pd.Series(y_train['Total_Recovered_Cases'])
            q = pd.Series(y_test['Total_Recovered_Cases'])
            model.fit(X_train[:1200], p[:1200])
            prd = model.predict(X_test[:1200])
            # print(mean_absolute_error(q, prd))
            # print(mean_squared_error(q, prd))
            # pd.np.sqrt(mean_squared_error(q, prd))
            b= r2_score(q[:1200], prd)
        return render_template('Model_Recoveries.html', msg="Error", b=b)
        return render_template("Model_Recoveries.html")

    return render_template(("Model_Recoveries.html"))




@app.route('/cases_pred',methods = ['GET','POST'])
def cases_pred():
    if request.method == "POST":
        a = int(request.form['0'])
        # bb=int(request.form['1'])
        print('ghdhsfh')
        b = int(request.form['1'])
        c = int(request.form['2'])
        d = int(request.form['3'])
        e = int(request.form['4'])
        f = int(request.form['5'])
        g = int(request.form['6'])
        print('skjksd')
        model = LinearRegression()
        print('KLKLKLKKLKLK')
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        model.fit(X_train, y_train)
        print('kjkjkjjk')
        ca = [a, b, c, d, e, f, g]
        predictions = model.predict([ca])
        print(predictions)
        # prediction = model.predict(X_test)
        # g=[a,b,c,d,e,f]
        return render_template('cases_pred.html', message ="cases" , c=predictions)
    return render_template('cases_pred.html')


@app.route('/deaths_pred',methods=['GET','POST'])
def deaths_pred():

    if request.method == "POST":
        a=int(request.form['7'])
        # bb=int(request.form['1'])
        b=int(request.form['8'])
        c=int(request.form['9'])
        d=int(request.form['10'])
        e=int(request.form['11'])
        f=int(request.form['12'])
        g =int(request.form['13'])
        model = LinearRegression()
        # print('KLKLKLKKLKLK')
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        model.fit(X_train,y_train)
        # print('kjkjkjjk')
        ca =[a,b,c,d,e,f,g]
        predictions = model.predict([ca])
        # prediction = model.predict(X_test)
        # g=[a,b,c,d,e,f]
        return render_template('deaths_pred.html', message ="cases" ,b = predictions)

    return render_template('deaths_pred.html')
#
@app.route('/rec_pred)',methods=['GET','POST'])
def rec_pred():
    if request.method=="POST":
        h = int(request.form['7'])
        # bb=int(request.form['1'])
        i = int(request.form['8'])
        j = int(request.form['9'])
        k = int(request.form['10'])
        l = int(request.form['11'])
        m = int(request.form['12'])
        n = int(request.form['13'])
        model = LinearRegression()
        # print('KLKLKLKKLKLK')
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        model.fit(X_train, y_train)
        # print('kjkjkjjk')
        de = [h, i, j, k, l, m, n]
        predictions = model.predict([de])
        # prediction = model.predict(X_test)
        # g=[a,b,c,d,e,f]
        return render_template('rec_pred.html', message="deaths", a=predictions)



    return render_template("rec_pred.html")

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == "POST":
        un=request.form['name']
        em=request.form['Email']
        pw=request.form['psw']
        cpw=request.form['cpsw']
        if pw == cpw:
            sql= "SELECT * FROM covid_reg"
            cur = mydb.cursor()
            cur.execute(sql)
            all_emails=cur.fetchall()
            mydb.commit()
            all_email=[i[2] for i in all_emails]
            print(all_email)
            if em in all_email:
                return render_template('register.html',msg='User already exists')
            else:
                sql="INSERT INTO covid_reg(name,Email,psw,cpsw) values(%s,%s,%s,%s)"
                val = (un,em,pw,cpw)
                cur=mydb.cursor()
                cur.execute(sql,val)
                mydb.commit()
                cur.close()
                return render_template('register.html', msg = 'Success')
        else:
            return render_template('register.html',msg='password_repeat')

    return render_template('register.html')

@app.route('/login',methods=['POST','GET'])
def login():
    print('kbbbbbbbbbbbbbbk')
    if request.method == "POST":
        em = request.form['Email']
        psw= request.form['psw']
        cur=mydb.cursor()
        print('kjhvkjbdvbkjdb')
        sql = "SELECT * FROM covid_reg where Email = '%s' and psw = '%s'" %(em,psw)
        cur.execute(sql)

        results = cur.fetchall()
        mydb.commit()
        cur.close()
        print(results)
        if len(results) >0:
            print('jhgjhgjhgjgh')
            return render_template('load_data.html', msg="Login Sucessful")

        else:
            print('mouli')
            return render_template('login.html', msg="Fail")


    return render_template('login.html')














if __name__=='__main__':
    app.run(debug=True)