from turtle import ondrag
from django.db import models
from .marcas import Marca
from .unidades import Unidad
from .categorias import Categoria
from ...compras.models.compras import Compra


class Producto(models.Model):
    marca = models.ForeignKey(Marca, on_delete=models.DO_NOTHING)
    compra = models.ForeignKey(Compra, on_delete=models.DO_NOTHING)
    unidad = models.ForeignKey(Unidad, on_delete=models.DO_NOTHING)
    categoria = models.ForeignKey(Categoria, on_delete=models.DO_NOTHING)
    nombre = models.CharField(max_length=200)
    descripcion = models.CharField(max_length=255, null=True, blank=True)
    precio_compra = models.FloatField(null=True, blank=True)
    precio_venta = models.FloatField(null=True, blank=True)
    utilidad = models.FloatField(null=True, blank=True)
    existencia = models.FloatField(null=True, blank=True)
    lista_precios = models.JSONField()
    archivo = models.FileField(upload_to='media')
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    def __str__(self):
        return self.nombre
