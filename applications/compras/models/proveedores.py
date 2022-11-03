from django.db import models

class Proveedor(models.Model):
    nombre = models.CharField(max_length=128)
    descripcion = models.CharField(max_length=128, blank=True, null= True)
    nit = models.CharField(max_length=128, blank=True, null= True)
    direccion = models.CharField(max_length=128, blank=True, null= True)
    correo = models.CharField(max_length=128, blank=True, null= True)
    telefono = models.CharField(max_length=128, blank=True, null= True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    class Meta:
        verbose_name = 'Proveedor'
        verbose_name_plural = 'Proveedores'

    def __str__(self):
        return self.nombre
