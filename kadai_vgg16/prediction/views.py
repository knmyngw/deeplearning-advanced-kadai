from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from io import BytesIO
import os
import math


def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)
            results = model.predict(img_array)
            results= decode_predictions(results)
            prediction = results[0][0][1]
            category = []
            prob = []
            for result in results[0]:
                category.append(result[1])
                prob.append(round(result[2]*100, 1))


            img_data = request.POST.get('img_data')
        return render(request, 'home.html', {'form': form,
                                             'results': results,
                                             'category': category,
                                             'prob': prob,
                                             'prediction': prediction,
                                             'img_data': img_data})
    else:
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
