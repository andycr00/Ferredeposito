from django.db import models

class Cliente(models.Model):
    razon_social = models.CharField(max_length=255)
    descripcion = models.CharField(max_length=255, null=True)
    nit = models.CharField(max_length=30, null=True, blank=True)
    correo = models.CharField(max_length=100, null=True)
    direccion = models.CharField(max_length=100, null=True)
    telefono = models.CharField(max_length=100, null=True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    def __str__(self):
        return self.razon_social
