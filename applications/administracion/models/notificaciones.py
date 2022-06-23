from django.db import models

class Notificacion(models.Model):
    titulo = models.CharField(max_length=128)
    descripcion = models.CharField(max_length=128)
    leido = models.BooleanField(default=False, null=True, blank=True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    def __str__(self):
        return self.titulo
