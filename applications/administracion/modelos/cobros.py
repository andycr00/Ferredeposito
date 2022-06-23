from django.db import models
# from ...ventas.models import Venta
# import applications.ventas.models as V

class Cobro(models.Model):
    # venta = models.ForeignKey(Venta, on_delete=models.DO_NOTHING)
    descripcion = models.CharField(max_length=300)
    fecha_pago = models.DateTimeField(null=True, blank=True)
    valor = models.FloatField(null=True, blank=True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    def __str__(self):
        return self.venta
