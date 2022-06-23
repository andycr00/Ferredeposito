from django.contrib import admin
from ..models import *


class ComprasAdmin(admin.ModelAdmin):
    list_display = ['id', 'no_factura', 'fecha_movimiento', 'fecha_factura',]
    search_fields = ['no_factura']
    raw_id_fields = ['proveedor']

class ProveedoresAdmin(admin.ModelAdmin):
    list_display = ['id', 'nombre', 'nit', 'telefono', 'correo', ]
    search_fields = ['nit', 'nombre']

admin.site.register(Compra, ComprasAdmin)
admin.site.register(Proveedor, ProveedoresAdmin)
