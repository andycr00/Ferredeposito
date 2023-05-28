from django.contrib import admin
from ..models import *

class ClientesAdmin(admin.ModelAdmin):
    list_display = ['id', 'razon_social', 'nit', 'telefono', 'correo']
    search_fields = ['razon_social', 'nit', 'telefono']

class CobrosAdmin(admin.ModelAdmin):
    list_display = ['id', 'venta', 'valor_cobro', 'fecha_pago']
    search_fields = ['descripcion', 'fecha_pago', 'valor' ]
    # raw_id_fields = ['venta']

    def valor_cobro(self, obj):
        return '${:,.2f}'.format(obj.valor) if obj.valor else ""

class ProductoVentaAdmin(admin.ModelAdmin):
    list_display = ['id', 'producto', 'venta', 'valor_unit', 'cantidad', 'valor_tot']
    search_fields = ['venta', 'producto',]
    # raw_id_fields = ['venta', 'producto']

    def valor_unit(self, obj):
        return '${:,.2f}'.format(obj.valor_unitario) if obj.valor_unitario else ""

    def valor_tot(self, obj):
        return '${:,.2f}'.format(obj.valor_total) if obj.valor_total else ""

class VentasAdmin(admin.ModelAdmin):
    list_display = ['id', 'cliente', 'valor_tot', 'fecha', 'forma_pago']
    search_fields = ['cliente__razon_social', 'fecha',]

    # raw_id_fields = ['venta', 'producto']

    def valor_tot(self, obj):
        return '${:,.2f}'.format(obj.valor_total) if obj.valor_total else ""

admin.site.register(Cliente, ClientesAdmin)
admin.site.register(Cobro, CobrosAdmin)
admin.site.register(ProductoVenta, ProductoVentaAdmin)
admin.site.register(Venta, VentasAdmin)