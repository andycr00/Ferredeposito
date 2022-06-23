from django.shortcuts import render
from django.views.generic import TemplateView, ListView

class Prueba(TemplateView):
    template_name = 'ventas/prueba.html'


class PruebaList(ListView):
    template_name = "ventas/lista.html"
    context_object_name = 'listaNumeros'
    queryset = ['1', '10', '20', '30']
    