from django.db import models

class CierreCaja(models.Model):
    fecha = models.DateField(auto_now_add=True ,null=True, blank=True)
    observacion = models.CharField(max_length=300)
    valor = models.FloatField()
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True, editable=False)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    def __str__(self):
        return self.fecha
