from django.db import models

class Categoria(models.Model):
    nombre = models.CharField(max_length=128, verbose_name='Categoria')

    def __str__(self):
        return self.nombre
