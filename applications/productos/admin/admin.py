from django.contrib import admin
from ..models import *


class CategoriasAdmin(admin.ModelAdmin):
    list_display = ['id', 'nombre',]
    search_fields = ['nombre']

class MarcasAdmin(admin.ModelAdmin):
    list_display = ['id', 'nombre',]
    search_fields = ['nombre']

class UnidadesAdmin(admin.ModelAdmin):
    list_display = ['id', 'nombre',]
    search_fields = ['nombre']

class ProductoAdmin(admin.ModelAdmin):
    list_display = ['id', 'nombre', 'precio_compra', 'precio_venta', 'existencia','utilidad']
    search_fields = ['nombre', 'categoria', 'marca']

# class DeudaAdmin(admin.ModelAdmin):
#     list_display = ['id', 'compra', 'saldo','fecha_inicio', 'fecha_fin']
#     search_fields = ['compra', 'fecha_inicio']
#     raw_id_fields = ['compra']

admin.site.register(Categoria, CategoriasAdmin)
admin.site.register(Marca, MarcasAdmin)
admin.site.register(Unidad, UnidadesAdmin)
admin.site.register(Producto, ProductoAdmin)