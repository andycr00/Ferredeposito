from django.db import models

class Unidad(models.Model):
    nombre = models.CharField(max_length=128, verbose_name='Unidades')

    class Meta:
        verbose_name = 'Unidad'
        verbose_name_plural = 'Unidades'

    def __str__(self):
        return self.nombre
