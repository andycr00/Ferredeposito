from django.db import models
from .cliente import Cliente
from ...administracion.modelos.usuarios import Usuario
from ...productos.models.productos import Producto

PAYMENT_CHOICES = (
    ("EFECTIVO", "Efectivo"),
    ("TRANSFERENCIA", "Transferencia"),
    ("NEQUI", "Nequi"),
    ("DAVIPLATA", "Daviplata"),
    ("DATAFONO", "Datafono"),
)

class Venta(models.Model):
    cliente = models.ForeignKey(Cliente, on_delete=models.DO_NOTHING)
    usuario = models.ForeignKey(Usuario, on_delete=models.DO_NOTHING)
    producto = models.ManyToManyField(Producto, through='ProductoVenta')
    fecha = models.DateTimeField(auto_now_add=True, editable=True)
    valor_total = models.FloatField(default=0, null=False)
    valor_antes_impuestos = models.FloatField(default=0, null=False)
    impuestos = models.JSONField()
    total_impuestos = models.FloatField(default=0, null=True, blank=True)
    observacion = models.CharField(max_length=255, null=True)
    forma_pago = models.CharField(max_length=200, choices=PAYMENT_CHOICES, default="EFECTIVO", null=True, blank=True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True, editable=False)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)


class ProductoVenta(models.Model):
    venta = models.ForeignKey(Venta, on_delete=models.CASCADE)
    producto = models.ForeignKey(Producto, on_delete=models.CASCADE)
    cantidad = models.FloatField(default=1, null=True, blank=True)
    valor_unitario = models.FloatField(default=0, null=True, blank=True)
    valor_total = models.FloatField(default=0, null=True, blank=True)
    utilidad = models.FloatField(default=0, null=True, blank=True)
    creado_en = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    actualizado_en = models.DateTimeField(auto_now=True, null=True, blank=True)