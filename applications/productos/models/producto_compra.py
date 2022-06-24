from django.db import models
from .productos import Producto
from .unidades import Unidad
from .marcas import Marca
from ...compras.models.compras import Compra

class ProductoCompra(models.Model):
    producto = models.ForeignKey(Producto, on_delete=models.DO_NOTHING)
    unidad = models.ForeignKey(Unidad, on_delete=models.DO_NOTHING)
    compra = models.ForeignKey(Compra, on_delete=models.DO_NOTHING)
    marca = models.ForeignKey(Marca, on_delete=models.DO_NOTHING)
    precio_compra = models.FloatField(null=True, blank=True)
    precio_venta = models.FloatField(null=True, blank=True)
    descripcion = models.CharField(max_length=300 ,null=True, blank=True)
    cantidad = models.FloatField(null=True, blank=True)
    total_impuestos = models.FloatField(null=True, blank=True)
    descuento = models.FloatField(null=True, blank=True)
    utilidad = models.FloatField(null=True, blank=True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    class Meta:
        verbose_name = 'Producto Compra'
        verbose_name_plural = 'Productos Compras'

    def __str__(self):
        return self.producto