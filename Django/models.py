from django.db import models

class signupdb(models.Model) :
    firstname = models.CharField(max_length=30)
    lastname = models.CharField(max_length=30)
    dob = models.CharField(max_length=20)
    email = models.EmailField(max_length=50)
    mobileno = models.CharField(max_length=20)
    passwrd = models.CharField(max_length=30)
    confirmpass = models.CharField(max_length=30)
    class Meta :
        db_table : 'signupdb'

class logindb(models.Model) :
    firstname = models.CharField(max_length=30)
    lastname = models.CharField(max_length=30)
    email = models.EmailField(max_length=50)
    passwrd = models.CharField(max_length=20)
    class Meta :
        db_table : 'logindb'

class detailsmodel(models.Model) :
    firstname = models.CharField(max_length=30)
    lastname = models.CharField(max_length=30)
    age = models.IntegerField()
    annualincome = models.IntegerField()
    spendingscore = models.IntegerField()

class filesmodel(models.Model) :
    uploadfile = models.FileField(upload_to='E:\6th_Semester_Mini_Project\Mini_Project\Front_End\Django')
