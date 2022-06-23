from django.contrib import admin
from ..models import *


class UsuariosAdmin(admin.ModelAdmin):
    list_display = ['id', 'nombres', 'apellidos', 'correo']
    search_fields = ['nombres']

class PagoAdmin(admin.ModelAdmin):
    list_display = ['id', 'titulo', 'fecha', 'valor']
    search_fields = ['fecha', 'titulo']
    # raw_id_fields = ['deuda']

class DeudaAdmin(admin.ModelAdmin):
    list_display = ['id', 'compra', 'saldo','fecha_inicio', 'fecha_fin']
    search_fields = ['compra', 'fecha_inicio']
    # raw_id_fields = ['compra']

admin.site.register(Usuario, UsuariosAdmin)
admin.site.register(Pago, PagoAdmin)
admin.site.register(Deuda, DeudaAdmin)
# admin.site.register(Usuario, UsuariosAdmin)