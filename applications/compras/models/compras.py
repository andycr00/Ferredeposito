from django.db import models
from .proveedores import Proveedor

PAYMENT_CHOICES = (
    ("EFECTIVO", "Efectivo"),
    ("TRANSFERENCIA", "Transferencia"),
    ("NEQUI", "Nequi"),
    ("DAVIPLATA", "Daviplata"),
    ("DATAFONO", "Datafono"),
)

class Compra(models.Model):
    proveedor = models.ForeignKey(Proveedor, on_delete=models.DO_NOTHING)
    fecha_factura = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    fecha_movimiento = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    no_factura = models.CharField(max_length= 100,blank=True, null= True)
    forma_pago = models.CharField(max_length=200, choices=PAYMENT_CHOICES, default="EFECTIVO", null=True, blank=True)
    valor_compra = models.FloatField(default=0,blank=True, null=True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True, editable=False)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)

    def __str__ (self):
        return self.no_factura