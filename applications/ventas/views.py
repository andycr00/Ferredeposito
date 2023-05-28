from django.shortcuts import render 
from django.views.generic import TemplateView, ListView, UpdateView
from .models import Cliente


class Prueba(TemplateView):
    template_name = 'ventas/index.html'


class PruebaList(ListView):
    template_name = "ventas/lista.html"
    context_object_name = 'listaNumeros'
    queryset = ['1', '10', '20', '30']


class ClienteUpdateView(UpdateView):
    model = Cliente
    template_name = "ventas/lista.html"
    fields = ['razon_social', 'nit', 'correo', 'direccion']
    def get_success_url(self):
        return render(request,'prod', kwargs={'id': self.object.id})


class ProdVentaList(ListView):
    model = Cliente
    template_name = "ventas/prueba.html"
    context_object_name = 'listaN'
