from django.shortcuts import render,redirect
from django import template
from django.template import loader
from django.http import HttpResponse
from django.db import connections,transaction
from django.core.files.storage import FileSystemStorage
from Django import forms
from Django import models
from Django import settings
from django.contrib import messages
import pickle

def homepage(request) :
    template = loader.get_template('HomePage.html')
    return HttpResponse(template.render())

def services(request) :
    template2 = loader.get_template('Services.html')
    return HttpResponse(template2.render())

def signuppage(request) :
    form = forms.signupform(request.POST or None)
    if form.is_valid() :
        form.save()
        cursor = connections['default'].cursor()
        #_sql = "INSERT INTO login(login.firstname, login.lastname, login.email, login.passwrd) SELECT DISTINCT signup.firstname, signup.lastname, signup.email, signup.passwrd FROM django_logindb login, django_signupdb signup WHERE ((login.firstname != signup.firstname) AND (login.lastname != signupup.lastname))"
        cursor.execute("TRUNCATE TABLE django_logindb")
        cursor.execute("INSERT INTO django_logindb(firstname, lastname, email, passwrd) SELECT DISTINCT firstname, lastname, email, passwrd FROM django_signupdb")
        transaction.commit()

    context = {'form': form}
    return render(request,'SignupPage.html',context)

def loginauth(request) :
    if request.method == 'POST' :
        if request.POST['emailid'] and request.POST['passwrd'] :
            email = request.POST['emailid']
            passwrd = request.POST['passwrd']
            cursor = connections['default'].cursor()
            sql = "SELECT firstname, lastname FROM django_logindb WHERE (email = '{}' AND passwrd = '{}')".format(email, passwrd)
            if (cursor.execute(sql)):
                transaction.commit()
                return redirect('servicesview')
            messages.error(request, "Invalid e-mail or password")
            return render(request, 'LoginPage.html')
        else :
            return render(request, 'LoginPage.html')
    else :
        return render(request, 'LoginPage.html')

def getpredictions(age,annualincome,spendingscore) :
    model = pickle.load(open("logreg_model.sav", "rb"))
    scaled = pickle.load(open("scaled_model.sav", "rb"))
    predict = model.predict(scaled.transform([[age,annualincome,spendingscore]]))
    return predict

def resultdetails(request) :
    if request.method == 'POST' :
        if request.POST['firstname'] and request.POST['lastname'] and request.POST['age'] and request.POST['annualincome'] and request.POST['spendingscore'] :
            firstname = request.POST['firstname']
            lastname = request.POST['lastname']
            age = request.POST['age']
            annualincome = request.POST['annualincome']
            spendingscore = request.POST['spendingscore']
            predict = getpredictions(age, annualincome, spendingscore)
            if predict == 1 :
                result = "The Customer has the possibility of buying the product :)"
                message1 = "Prediction : {}".format(result)
            else :
                result = "The Customer does not have the possibility of buying the product :("
                message1 = "Prediction : {}".format(result)
            context = {
                'firstname': firstname,
                'lastname': lastname,
                'age': age,
                'annualincome': annualincome,
                'spendingscore': spendingscore,
                'message1': message1
            }
            return render(request, 'AnalyticsDetailsResults.html', context)
    else :
        return render(request, 'AnalysisDetails.html')

def resultsfile(request) :
    if request.method == 'POST' :
        file = request.FILES['uploadfiles']
        fs = FileSystemStorage()
        fs.save(file.name, file)
        context = {'data' : kmeansmodel(request,file), 'iteration': temp_columns(request,file), 'temp1': temp_row1(request,file), 'temp2': temp_row2(request,file), 'temp3': temp_row3(request,file), 'temp4': temp_row4(request,file), 'temp5': temp_row5(request,file), 'temp6': temp_row6(request,file), 'temp7': temp_row7(request, file), 'temp8': temp_row8(request, file)}
        return render(request, 'AnalyticsFilesResults.html', context)
    else :
        return render(request, 'AnalyticsFiles.html')

def kmeansmodel(request,file) :
    #Importing the Libraries
    import io,urllib,base64
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import random

    #Importing the Dataset
    dataset = pd.read_csv('datasets/%s'%(file))
    x = dataset.iloc[:, [5,6]].values

    #Using the elbow method to find the optimal number of clusters
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    length = len(wcss)
    diff_wcss = []
    for i in range(0, length):
        if i == length - 1:
            break
        else:
            difference = wcss[i] - wcss[i + 1]
            diff_wcss.append(difference)
    avg = sum(diff_wcss) / len(diff_wcss)

    #Training the K-Means model on the dataset
    for i in range(0, len(diff_wcss)):
        if diff_wcss[i] < avg:
            cluster = i + 1
            break
    kmeans = KMeans(n_clusters=cluster, init="k-means++", random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    # Visualising the clusters
    number_of_colors = cluster
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(cluster+1)]) for i in range(number_of_colors)]
    for i in range(0, len(color)):
        plt.scatter(x[y_kmeans == i, 0], x[y_kmeans == i, 1], s=100, c=color[i], label='Cluster %s' %(i+1))
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', label='Centroids')
    plt.title('Clusters of Customers')
    plt.xlabel('Annual Income(k$)')
    plt.ylabel('Spending Score(1 - 100)')
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    plt.close()
    return uri

def temp_columns(request,file):
    # Importing the Libraries
    import pandas as pd

    # Importing the Dataset
    dataset = pd.read_csv('datasets/%s' % (file))
    x = dataset.iloc[:, [5, 6]].values

    # Using the elbow method to find the optimal number of clusters
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    length = len(wcss)
    diff_wcss = []
    for i in range(0, length):
        if i == length - 1:
            break
        else:
            difference = wcss[i] - wcss[i + 1]
            diff_wcss.append(difference)
    avg = sum(diff_wcss) / len(diff_wcss)

    # Training the K-Means model on the dataset
    for i in range(0, len(diff_wcss)):
        if diff_wcss[i] < avg:
            cluster = i + 1
            break
    kmeans = KMeans(n_clusters=cluster, init="k-means++", random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    # Table Creation
    customer_id = dataset.iloc[:, 0].values
    customer_name = dataset.iloc[:, 1].values
    gender = dataset.iloc[:, 2].values
    customer_age = dataset.iloc[:, 3].values
    mobile_number = dataset.iloc[:, 4].values
    annual_income = dataset.iloc[:, 5].values
    spending_score = dataset.iloc[:, 6].values
    predict = y_kmeans
    id_dict = {}
    id_dict['customer_id'] = customer_id
    id_dict['customer_name'] = customer_name
    id_dict['gender'] = gender
    id_dict['age'] = customer_age
    id_dict['mobile_number'] = mobile_number
    id_dict['annualincome'] = annual_income
    id_dict['spendingscore'] = spending_score
    id_dict['cluster'] = predict
    columns = []
    avg = sum(id_dict['annualincome']) / len(id_dict['annualincome'])
    for i in range(len(id_dict['customer_id'])):
        if id_dict['spendingscore'][i] > 50 and id_dict['annualincome'][i] > avg:
            columns.append(id_dict['customer_id'][i])
    length_column = []
    for i in range(0, len(columns)) :
        length_column.append(i)
    return length_column

temp1 = []
temp2 = []
temp3 = []
temp4 = []
temp5 = []
temp6 = []
temp7 = []
temp8 = []

def temp_row1(request, file) :
    # Importing the Libraries
    import pandas as pd

    # Importing the Dataset
    dataset = pd.read_csv('datasets/%s' % (file))
    x = dataset.iloc[:, [5, 6]].values

    # Using the elbow method to find the optimal number of clusters
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    length = len(wcss)
    diff_wcss = []
    for i in range(0, length):
        if i == length - 1:
            break
        else:
            difference = wcss[i] - wcss[i + 1]
            diff_wcss.append(difference)
    avg = sum(diff_wcss) / len(diff_wcss)

    # Training the K-Means model on the dataset
    for i in range(0, len(diff_wcss)):
        if diff_wcss[i] < avg:
            cluster = i + 1
            break
    kmeans = KMeans(n_clusters=cluster, init="k-means++", random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    #Table Creation
    customer_id = dataset.iloc[:, 0].values
    customer_name = dataset.iloc[:, 1].values
    gender = dataset.iloc[:, 2].values
    customer_age = dataset.iloc[:, 3].values
    mobile_number = dataset.iloc[:, 4].values
    annual_income = dataset.iloc[:, 5].values
    spending_score = dataset.iloc[:, 6].values
    predict = y_kmeans
    id_dict = {}
    id_dict['customer_id'] = customer_id
    id_dict['customer_name'] = customer_name
    id_dict['gender'] = gender
    id_dict['age'] = customer_age
    id_dict['mobile_number'] = mobile_number
    id_dict['annualincome'] = annual_income
    id_dict['spendingscore'] = spending_score
    id_dict['cluster'] = predict
    avg = sum(id_dict['annualincome'])/len(id_dict['annualincome'])
    for i in range(len(id_dict['customer_id'])) :
        if id_dict['spendingscore'][i]>50 and id_dict['annualincome'][i]>avg :
            temp1.append(id_dict['customer_id'][i])
    return temp1

def temp_row2(request, file) :
    # Importing the Libraries
    import pandas as pd

    # Importing the Dataset
    dataset = pd.read_csv('datasets/%s' % (file))
    x = dataset.iloc[:, [5, 6]].values

    # Using the elbow method to find the optimal number of clusters
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    length = len(wcss)
    diff_wcss = []
    for i in range(0, length):
        if i == length - 1:
            break
        else:
            difference = wcss[i] - wcss[i + 1]
            diff_wcss.append(difference)
    avg = sum(diff_wcss) / len(diff_wcss)

    # Training the K-Means model on the dataset
    for i in range(0, len(diff_wcss)):
        if diff_wcss[i] < avg:
            cluster = i + 1
            break
    kmeans = KMeans(n_clusters=cluster, init="k-means++", random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    #Table Creation
    customer_id = dataset.iloc[:, 0].values
    customer_name = dataset.iloc[:, 1].values
    gender = dataset.iloc[:, 2].values
    customer_age = dataset.iloc[:, 3].values
    mobile_number = dataset.iloc[:, 4].values
    annual_income = dataset.iloc[:, 5].values
    spending_score = dataset.iloc[:, 6].values
    predict = y_kmeans
    id_dict = {}
    id_dict['customer_id'] = customer_id
    id_dict['customer_name'] = customer_name
    id_dict['gender'] = gender
    id_dict['age'] = customer_age
    id_dict['mobile_number'] = mobile_number
    id_dict['annualincome'] = annual_income
    id_dict['spendingscore'] = spending_score
    id_dict['cluster'] = predict
    avg = sum(id_dict['annualincome'])/len(id_dict['annualincome'])
    for i in range(len(id_dict['customer_id'])) :
        if id_dict['spendingscore'][i]>50 and id_dict['annualincome'][i]>avg :
            temp2.append(id_dict['customer_name'][i])
    return temp2

def temp_row3(request, file) :
    # Importing the Libraries
    import pandas as pd

    # Importing the Dataset
    dataset = pd.read_csv('datasets/%s' % (file))
    x = dataset.iloc[:, [5, 6]].values

    # Using the elbow method to find the optimal number of clusters
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    length = len(wcss)
    diff_wcss = []
    for i in range(0, length):
        if i == length - 1:
            break
        else:
            difference = wcss[i] - wcss[i + 1]
            diff_wcss.append(difference)
    avg = sum(diff_wcss) / len(diff_wcss)

    # Training the K-Means model on the dataset
    for i in range(0, len(diff_wcss)):
        if diff_wcss[i] < avg:
            cluster = i + 1
            break
    kmeans = KMeans(n_clusters=cluster, init="k-means++", random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    #Table Creation
    customer_id = dataset.iloc[:, 0].values
    customer_name = dataset.iloc[:, 1].values
    gender = dataset.iloc[:, 2].values
    customer_age = dataset.iloc[:, 3].values
    mobile_number = dataset.iloc[:, 4].values
    annual_income = dataset.iloc[:, 5].values
    spending_score = dataset.iloc[:, 6].values
    predict = y_kmeans
    id_dict = {}
    id_dict['customer_id'] = customer_id
    id_dict['customer_name'] = customer_name
    id_dict['gender'] = gender
    id_dict['age'] = customer_age
    id_dict['mobile_number'] = mobile_number
    id_dict['annualincome'] = annual_income
    id_dict['spendingscore'] = spending_score
    id_dict['cluster'] = predict
    avg = sum(id_dict['annualincome'])/len(id_dict['annualincome'])
    for i in range(len(id_dict['customer_id'])) :
        if id_dict['spendingscore'][i]>50 and id_dict['annualincome'][i]>avg :
            temp3.append(id_dict['gender'][i])
    return temp3

def temp_row4(request, file) :
    # Importing the Libraries
    import pandas as pd

    # Importing the Dataset
    dataset = pd.read_csv('datasets/%s' % (file))
    x = dataset.iloc[:, [5, 6]].values

    # Using the elbow method to find the optimal number of clusters
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    length = len(wcss)
    diff_wcss = []
    for i in range(0, length):
        if i == length - 1:
            break
        else:
            difference = wcss[i] - wcss[i + 1]
            diff_wcss.append(difference)
    avg = sum(diff_wcss) / len(diff_wcss)

    # Training the K-Means model on the dataset
    for i in range(0, len(diff_wcss)):
        if diff_wcss[i] < avg:
            cluster = i + 1
            break
    kmeans = KMeans(n_clusters=cluster, init="k-means++", random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    #Table Creation
    customer_id = dataset.iloc[:, 0].values
    customer_name = dataset.iloc[:, 1].values
    gender = dataset.iloc[:, 2].values
    customer_age = dataset.iloc[:, 3].values
    mobile_number = dataset.iloc[:, 4].values
    annual_income = dataset.iloc[:, 5].values
    spending_score = dataset.iloc[:, 6].values
    predict = y_kmeans
    id_dict = {}
    id_dict['customer_id'] = customer_id
    id_dict['customer_name'] = customer_name
    id_dict['gender'] = gender
    id_dict['age'] = customer_age
    id_dict['mobile_number'] = mobile_number
    id_dict['annualincome'] = annual_income
    id_dict['spendingscore'] = spending_score
    id_dict['cluster'] = predict
    avg = sum(id_dict['annualincome'])/len(id_dict['annualincome'])
    for i in range(len(id_dict['customer_id'])) :
        if id_dict['spendingscore'][i]>50 and id_dict['annualincome'][i]>avg :
            temp4.append(id_dict['age'][i])
    return temp4

def temp_row5(request, file) :
    # Importing the Libraries
    import pandas as pd

    # Importing the Dataset
    dataset = pd.read_csv('datasets/%s' % (file))
    x = dataset.iloc[:, [5, 6]].values

    # Using the elbow method to find the optimal number of clusters
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    length = len(wcss)
    diff_wcss = []
    for i in range(0, length):
        if i == length - 1:
            break
        else:
            difference = wcss[i] - wcss[i + 1]
            diff_wcss.append(difference)
    avg = sum(diff_wcss) / len(diff_wcss)

    # Training the K-Means model on the dataset
    for i in range(0, len(diff_wcss)):
        if diff_wcss[i] < avg:
            cluster = i + 1
            break
    kmeans = KMeans(n_clusters=cluster, init="k-means++", random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    #Table Creation
    customer_id = dataset.iloc[:, 0].values
    customer_name = dataset.iloc[:, 1].values
    gender = dataset.iloc[:, 2].values
    customer_age = dataset.iloc[:, 3].values
    mobile_number = dataset.iloc[:, 4].values
    annual_income = dataset.iloc[:, 5].values
    spending_score = dataset.iloc[:, 6].values
    predict = y_kmeans
    id_dict = {}
    id_dict['customer_id'] = customer_id
    id_dict['customer_name'] = customer_name
    id_dict['gender'] = gender
    id_dict['age'] = customer_age
    id_dict['mobile_number'] = mobile_number
    id_dict['annualincome'] = annual_income
    id_dict['spendingscore'] = spending_score
    id_dict['cluster'] = predict
    avg = sum(id_dict['annualincome'])/len(id_dict['annualincome'])
    for i in range(len(id_dict['customer_id'])) :
        if id_dict['spendingscore'][i]>50 and id_dict['annualincome'][i]>avg :
            temp5.append(id_dict['mobile_number'][i])
    return temp5

def temp_row6(request, file) :
    # Importing the Libraries
    import pandas as pd

    # Importing the Dataset
    dataset = pd.read_csv('datasets/%s' % (file))
    x = dataset.iloc[:, [5, 6]].values

    # Using the elbow method to find the optimal number of clusters
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    length = len(wcss)
    diff_wcss = []
    for i in range(0, length):
        if i == length - 1:
            break
        else:
            difference = wcss[i] - wcss[i + 1]
            diff_wcss.append(difference)
    avg = sum(diff_wcss) / len(diff_wcss)

    # Training the K-Means model on the dataset
    for i in range(0, len(diff_wcss)):
        if diff_wcss[i] < avg:
            cluster = i + 1
            break
    kmeans = KMeans(n_clusters=cluster, init="k-means++", random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    #Table Creation
    customer_id = dataset.iloc[:, 0].values
    customer_name = dataset.iloc[:, 1].values
    gender = dataset.iloc[:, 2].values
    customer_age = dataset.iloc[:, 3].values
    mobile_number = dataset.iloc[:, 4].values
    annual_income = dataset.iloc[:, 5].values
    spending_score = dataset.iloc[:, 6].values
    predict = y_kmeans
    id_dict = {}
    id_dict['customer_id'] = customer_id
    id_dict['customer_name'] = customer_name
    id_dict['gender'] = gender
    id_dict['age'] = customer_age
    id_dict['mobile_number'] = mobile_number
    id_dict['annualincome'] = annual_income
    id_dict['spendingscore'] = spending_score
    id_dict['cluster'] = predict
    avg = sum(id_dict['annualincome'])/len(id_dict['annualincome'])
    for i in range(len(id_dict['customer_id'])) :
        if id_dict['spendingscore'][i]>50 and id_dict['annualincome'][i]>avg :
            temp6.append(id_dict['annualincome'][i])
    return temp6

def temp_row7(request, file) :
    # Importing the Libraries
    import pandas as pd

    # Importing the Dataset
    dataset = pd.read_csv('datasets/%s' % (file))
    x = dataset.iloc[:, [5, 6]].values

    # Using the elbow method to find the optimal number of clusters
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    length = len(wcss)
    diff_wcss = []
    for i in range(0, length):
        if i == length - 1:
            break
        else:
            difference = wcss[i] - wcss[i + 1]
            diff_wcss.append(difference)
    avg = sum(diff_wcss) / len(diff_wcss)

    # Training the K-Means model on the dataset
    for i in range(0, len(diff_wcss)):
        if diff_wcss[i] < avg:
            cluster = i + 1
            break
    kmeans = KMeans(n_clusters=cluster, init="k-means++", random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    #Table Creation
    customer_id = dataset.iloc[:, 0].values
    customer_name = dataset.iloc[:, 1].values
    gender = dataset.iloc[:, 2].values
    customer_age = dataset.iloc[:, 3].values
    mobile_number = dataset.iloc[:, 4].values
    annual_income = dataset.iloc[:, 5].values
    spending_score = dataset.iloc[:, 6].values
    predict = y_kmeans
    id_dict = {}
    id_dict['customer_id'] = customer_id
    id_dict['customer_name'] = customer_name
    id_dict['gender'] = gender
    id_dict['age'] = customer_age
    id_dict['mobile_number'] = mobile_number
    id_dict['annualincome'] = annual_income
    id_dict['spendingscore'] = spending_score
    id_dict['cluster'] = predict
    avg = sum(id_dict['annualincome'])/len(id_dict['annualincome'])
    for i in range(len(id_dict['customer_id'])) :
        if id_dict['spendingscore'][i]>50 and id_dict['annualincome'][i]>avg :
            temp7.append(id_dict['spendingscore'][i])
    return temp7

def temp_row8(request, file) :
    # Importing the Libraries
    import pandas as pd

    # Importing the Dataset
    dataset = pd.read_csv('datasets/%s' % (file))
    x = dataset.iloc[:, [5, 6]].values

    # Using the elbow method to find the optimal number of clusters
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    length = len(wcss)
    diff_wcss = []
    for i in range(0, length):
        if i == length - 1:
            break
        else:
            difference = wcss[i] - wcss[i + 1]
            diff_wcss.append(difference)
    avg = sum(diff_wcss) / len(diff_wcss)

    # Training the K-Means model on the dataset
    for i in range(0, len(diff_wcss)):
        if diff_wcss[i] < avg:
            cluster = i + 1
            break
    kmeans = KMeans(n_clusters=cluster, init="k-means++", random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    #Table Creation
    customer_id = dataset.iloc[:, 0].values
    customer_name = dataset.iloc[:, 1].values
    gender = dataset.iloc[:, 2].values
    customer_age = dataset.iloc[:, 3].values
    mobile_number = dataset.iloc[:, 4].values
    annual_income = dataset.iloc[:, 5].values
    spending_score = dataset.iloc[:, 6].values
    predict = y_kmeans
    id_dict = {}
    id_dict['customer_id'] = customer_id
    id_dict['customer_name'] = customer_name
    id_dict['gender'] = gender
    id_dict['age'] = customer_age
    id_dict['mobile_number'] = mobile_number
    id_dict['annualincome'] = annual_income
    id_dict['spendingscore'] = spending_score
    id_dict['cluster'] = predict
    avg = sum(id_dict['annualincome'])/len(id_dict['annualincome'])
    for i in range(len(id_dict['customer_id'])) :
        if id_dict['spendingscore'][i]>50 and id_dict['annualincome'][i]>avg :
            temp8.append(id_dict['cluster'][i])
    return temp8