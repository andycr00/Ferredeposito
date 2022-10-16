from django.db import models
from ...compras.models.compras import Compra

class Deuda(models.Model):
    compra = models.ForeignKey(Compra, on_delete=models.CASCADE)
    saldo = models.FloatField(null=True, blank=True)
    fecha_inicio = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    fecha_fin = models.DateTimeField(null=True, blank=True)
    estado = models.BooleanField(default=True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True, editable=False)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    class Meta:
        verbose_name = 'Cuenta por pagar'
        verbose_name_plural = 'Cuentas por pagar'

    def __str__(self):
        return str(self.pk)
