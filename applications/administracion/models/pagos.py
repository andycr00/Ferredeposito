from django.db import models
from django.forms import ValidationError
from .deudas import Deuda

class Pago(models.Model):
    deuda = models.ForeignKey(Deuda, on_delete=models.DO_NOTHING, verbose_name='Cuenta por pagar')
    titulo = models.CharField(max_length=200,)
    descripcion = models.CharField(max_length=200, null=True, blank=True)
    fecha = models.DateTimeField(null=True, blank=True)
    valor = models.FloatField(null=True, blank=True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True, editable=False)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    def __str__(self):
        return self.titulo

    def save(self, force_insert=False, force_update=False, *args, **kwargs):
        deuda = Deuda.objects.get(pk=self.deuda.id)
        if deuda.estado == False:
            raise ValidationError('-------No se puede agregar un pago a una deuda inactiva-------')

        if deuda.saldo <= 0:
            raise ValidationError('-------No se puede agregar un pago a una deuda con saldo 0-------')

        if self.valor > deuda.saldo:
            raise ValidationError('-------El valor pagado no puede ser mayor al saldo de la deuda-------')

        if deuda.saldo == self.valor:
            deuda.saldo = 0
            deuda.estado = False
            deuda.save()
        
        if deuda.saldo > self.valor:
            deuda.saldo = deuda.saldo - self.valor
            deuda.save()
        
        super().save(force_insert, force_update, *args, **kwargs)
