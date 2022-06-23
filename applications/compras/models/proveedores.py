from django.db import models

class Proveedor(models.Model):
    nombre = models.CharField(max_length=128)
    descripcion = models.CharField(max_length=128)
    nit = models.CharField(max_length=128)
    direccion = models.CharField(max_length=128)
    correo = models.CharField(max_length=128)
    telefono = models.CharField(max_length=128)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    def __str__(self):
        return self.nombre
