from django.contrib import admin
from ..models import *


class UsuariosAdmin(admin.ModelAdmin):
    list_display = ['id', 'nombres', 'apellidos', 'correo']
    search_fields = ['nombres']

class PagoAdmin(admin.ModelAdmin):
    list_display = ['id', 'titulo', 'fecha', 'valor_pago']
    search_fields = ['fecha', 'titulo']

    def valor_pago(self, obj):
        return '${:,.2f}'.format(obj.valor) if obj.valor else ""

class DeudaAdmin(admin.ModelAdmin):
    list_display = ['id', 'compra', 'valor_saldo','fecha_inicio', 'fecha_fin', 'estado']
    search_fields = ['compra__id', 'fecha_inicio']
    list_filter = ['estado']
    # raw_id_fields = ['compra']

    def valor_saldo(self, obj):
        return '${:,.2f}'.format(obj.saldo) if obj.saldo else ""

class NotificacionAdmin(admin.ModelAdmin):
    list_display = ['id', 'titulo', 'descripcion', 'leido',]
    list_filter = ['leido']

class CierreCajaAdmin(admin.ModelAdmin):
    list_display = ['id', 'fecha', 'cierre_ventas', 'cierre_gastos', 'cierre_compras','observacion']
    search_fields = ['fecha', ]

    def cierre_ventas(self, obj):
        return '${:,.2f}'.format(obj.valor_ventas) if obj.valor_ventas else ""

    def cierre_gastos(self, obj):
        return '${:,.2f}'.format(obj.valor_gastos) if obj.valor_gastos else ""

    def cierre_compras(self, obj):
        return '${:,.2f}'.format(obj.valor_compras) if obj.valor_compras else ""

class GastoAdmin(admin.ModelAdmin):
    list_display = ['id', 'titulo', 'valor_gasto','descripcion', 'fecha', 'tipo']
    search_fields = ['titulo', ]
    list_filter = ['tipo']

    def valor_gasto(self, obj):
        return '${:,.2f}'.format(obj.valor) if obj.valor else ""
    

admin.site.register(Usuario, UsuariosAdmin)
admin.site.register(Pago, PagoAdmin)
admin.site.register(Deuda, DeudaAdmin)
admin.site.register(Notificacion, NotificacionAdmin)
admin.site.register(CierreCaja, CierreCajaAdmin)
admin.site.register(Gasto, GastoAdmin)
