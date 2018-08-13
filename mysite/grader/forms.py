from django import forms

from .models import Question, Essay
class AnswerForm(forms.ModelForm):
    answer = forms.CharField(max_length=100000, widget=forms.Textarea(attrs={'rows': 5, 'placeholder': "What's on your mind?"}))

    class Meta:
        model = Essay
        fields = ['answer']

