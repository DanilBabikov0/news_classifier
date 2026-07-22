from django import forms

MODEL_CHOICES = [
    ("mlp", "MLP (Baseline)"),
    ("bert", "ruBERT"),
    ("bert_tiny", "ruBERT Tiny"),
]

class TextClassificationForm(forms.Form):
    text = forms.CharField(
        widget=forms.Textarea(attrs={
            "class": "form-control",
            "rows": 5,
            "placeholder": "Введите текст для классификации..."
        })
    )
    model = forms.ChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"})
    )