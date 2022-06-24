from django.db import models

class Marca(models.Model):
    nombre = models.CharField(max_length=128, verbose_name='Marcas')

    def __str__(self):
        return self.nombre
