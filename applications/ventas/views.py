from django.shortcuts import render
from django.views.generic import TemplateView, ListView

class Prueba(TemplateView):
    template_name = 'ventas/prueba.html'


class PruebaList(ListView):
    model = MODEL_NAME
    template_name: "str"