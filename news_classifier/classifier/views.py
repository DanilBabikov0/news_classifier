from django.shortcuts import render
from django.http import HttpResponse
from .forms import TextClassificationForm
from .utils import classify_text

def index(request):
    result = None
    if request.method == "POST":
        form = TextClassificationForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data["text"]
            model_type = form.cleaned_data["model"]

            result = classify_text(text, model_type)
    else:
        form = TextClassificationForm()

    return render(request, "classifier/index.html", {
        "form": form,
        "result": result
    })