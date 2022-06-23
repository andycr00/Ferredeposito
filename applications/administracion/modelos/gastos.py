from django.db import models

PAYMENT_CHOICES = (
    ("PERSONALES", "Personales"),
    ("FIJOS", "Fijos"),
)

class Gasto(models.Model):
    titulo = models.CharField(max_length=200, null=True, blank=True)
    descripcion = models.CharField(max_length=255, null=True, blank=True)
    valor = models.FloatField(null=True, blank=True)
    fecha = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    tipo = models.CharField(max_length=200, choices=PAYMENT_CHOICES, default="FIJOS", null=True, blank=True)

    def __str__(self):
        return self.titulo
