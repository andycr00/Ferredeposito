from django.db import models
from .ventas import Venta
from ...productos.models import Producto

class ProductoVenta(models.Model):
    venta = models.ForeignKey(Venta, on_delete=models.CASCADE)
    producto = models.ForeignKey(Producto, on_delete=models.CASCADE)
    cantidad = models.FloatField(default=1, null=True, blank=True)
    valor_unitario = models.FloatField(default=0, null=True, blank=True)
    valor_total = models.FloatField(default=0, null=True, blank=True)
    utilidad = models.FloatField(default=0, null=True, blank=True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)