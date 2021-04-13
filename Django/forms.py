from django import forms
from Django import models

class signupform(forms.ModelForm) :
    class Meta :
        model = models.signupdb
        fields = ["firstname", "lastname", "dob", "email", "mobileno", "passwrd", "confirmpass"]

class analysisdetails(forms.ModelForm) :
    class Meta :
        model = models.detailsmodel
        fields = ["firstname", "lastname", "age", "annualincome", "spendingscore"]

class analysisfiles(forms.Form) :
    uploadfile = forms.FileField(help_text='max. 20 megabytes')
    class Meta :
        model = models.filesmodel