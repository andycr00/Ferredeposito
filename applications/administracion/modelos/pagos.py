from django.db import models
from .deudas import Deuda

from applications.compras.models.compras import PAYMENT_CHOICES

PAYMENT_CHOICES = (
    ("ABONO", "Abono"),
    ("CANCELACION", "Cancelacion"),
)

class Pago(models.Model):
    deuda = models.ForeignKey(Deuda, on_delete=models.DO_NOTHING)
    titulo = models.CharField(max_length=200, null=True, blank=True)
    descripcion = models.CharField(max_length=200, null=True, blank=True)
    fecha = models.DateTimeField(null=True, blank=True)
    valor = models.FloatField(null=True, blank=True)
    tipo = models.CharField(max_length=200, choices=PAYMENT_CHOICES, default="ABONO", null=True, blank=True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True, editable=False)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    def __str__(self):
        return self.titulo
