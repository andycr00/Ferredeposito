from django.db import models

class CierreCaja(models.Model):
    fecha = models.DateField(auto_now_add=True ,null=True, blank=True, editable=True)
    observacion = models.CharField(max_length=300, null=True, blank=True)
    valor_ventas = models.FloatField(null=True, blank=True)
    valor_gastos = models.FloatField(null=True, blank=True)
    valor_compras = models.FloatField(null=True, blank=True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True, editable=False)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    def __str__(self):
        return str(self.pk)
