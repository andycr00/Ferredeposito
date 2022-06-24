from tkinter.tix import Tree
from turtle import ondrag
from django.db import models
from .marcas import Marca
from .unidades import Unidad
from .categorias import Categoria
from ...compras.models.compras import Compra


class Producto(models.Model):
    marca = models.ForeignKey(Marca, on_delete=models.DO_NOTHING, null=True, blank=True)
    compra = models.ForeignKey(Compra, on_delete=models.DO_NOTHING, null=True, blank=True)
    unidad = models.ForeignKey(Unidad, on_delete=models.DO_NOTHING, null=True, blank=True)
    categoria = models.ForeignKey(Categoria, on_delete=models.DO_NOTHING, null=True, blank=True)
    nombre = models.CharField(max_length=200)
    descripcion = models.CharField(max_length=255, null=True, blank=True)
    precio_compra = models.FloatField(default=0, null=True, blank=True)
    precio_venta = models.FloatField(null=True, blank=True)
    utilidad = models.FloatField(null=True, blank=True)
    utilidad_total = models.FloatField(null=True, blank=True)
    existencia = models.FloatField(null=True, blank=True)
    lista_precios = models.JSONField(null=True, blank=True)
    archivo = models.FileField(upload_to='media', null=True, blank=True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    def __str__(self):
        return self.nombre

    __original_costo = None
    __original_venta = None
    __original_utilidad = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__original_costo = self.precio_compra
        self.__original_venta = self.precio_venta
        self.__original_utilidad = self.utilidad

    def save(self, force_insert=False, force_update=False, *args, **kwargs):

        # Si el registro no existe
        if self.pk is None:
            # Y si el registro cuenta con costo y precio venta, pero no con la utilidad
            # se calcula la utilidad y se establece el valor de la utilidad total
            if self.precio_compra and self.precio_venta and not self.utilidad:
                valor_util = (self.precio_venta / self.precio_compra) - 1
                self.utilidad = valor_util * 100
                self.utilidad_total = valor_util * self.precio_compra
            # Si el registro cuenta con costo y utilidad, pero no con precio venta
            # Se calcula el precio de venta y se establece el valor de la utilidad total
            if self.precio_compra and self.utilidad and not self.precio_venta:
                self.precio_venta = self.precio_compra * \
                    (1 + (self.utilidad / 100))
                self.utilidad_total = (self.utilidad / 100 ) * self.precio_compra

        # Si hay un cambio en el valor del costo, el precio de venta no cambia y existe utilidad
        # Se calcula el precio de venta nuevamente y se establece el valor de la utilidad total
        if self.precio_compra != self.__original_costo and self.utilidad:
            self.precio_venta = self.precio_compra * \
                (1 + (self.utilidad / 100))
            self.utilidad_total = (self.utilidad / 100 ) * self.precio_compra
        
        # Si hay un cambio en el valor del precio venta, y existe la utilidad y el costo
        # Se calcula de nuevo la utilidad y se establece el valor de la utilidad total
        if self.precio_venta != self.__original_venta and self.utilidad and self.precio_compra:
            valor_util = (self.precio_venta / self.precio_compra) - 1
            self.utilidad = valor_util * 100
            self.utilidad_total = valor_util * self.precio_compra

        # Si cambia la utilidad, y existe el costo
        # Se calcula el precio venta y se establece el valor de la utilidad total
        if self.utilidad != self.__original_utilidad and self.precio_compra:
            self.precio_venta = self.precio_compra * \
                (1 + (self.utilidad / 100))
            self.utilidad_total = (self.utilidad / 100 ) * self.precio_compra

        super().save(force_insert, force_update, *args, **kwargs)
        
