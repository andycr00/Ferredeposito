from django.db import models

from .proveedores import Proveedor

from django.db.models.signals import post_save
from django.dispatch import receiver

# method for updating

PAYMENT_CHOICES = (
    ("CONTADO", "Contado"),
    ("CREDITO", "Credito"),
)

class Compra(models.Model):
    proveedor = models.ForeignKey(Proveedor, on_delete=models.DO_NOTHING)
    fecha_factura = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    fecha_movimiento = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    no_factura = models.CharField(max_length= 100,blank=True, null= True)
    forma_pago = models.CharField(max_length=200, choices=PAYMENT_CHOICES, default="CONTADO", null=True, blank=True)
    valor_compra = models.FloatField(blank=True, null=True)
    total_iva = models.FloatField(blank=True, null=True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True, editable=False)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    def __str__ (self):
        return str(self.pk)

@receiver(post_save, sender=Compra, dispatch_uid="crear_deuda")
def post_save_pedidos(sender, instance, created, *args, **kwargs):
    from applications.administracion.models import Deuda
    compra = instance

    if created:
        if compra.forma_pago == "CREDITO":
            Deuda.objects.create(
                compra=compra,
                saldo=compra.valor_compra,
            )

