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
    list_display = ['id', 'nombre', 'valor_compra', 'valor_venta', 'existencia','utilidad']
    search_fields = ['nombre',]
    list_filter = ['categoria__nombre', 'marca__nombre']
    # raw_id_fields = ['compra']

    def valor_compra(self, obj):
        return '${:,.2f}'.format(obj.precio_compra) if obj.precio_compra else ""
    
    def valor_venta(self, obj):
        return '${:,.2f}'.format(obj.precio_venta) if obj.precio_venta else ""

class ProductoCompraAdmin(admin.ModelAdmin):
    list_display = ['id', 'producto', 'valor_compra','valor_venta', 'cantidad']
    search_fields = ['producto__nombre', 'compra__id']
    # raw_id_fields = ['compra', 'producto', 'unidad', 'marca']

    def valor_compra(self, obj):
        return '${:,.2f}'.format(obj.precio_compra) if obj.precio_compra else ""
    
    def valor_venta(self, obj):
        return '${:,.2f}'.format(obj.precio_venta) if obj.precio_venta else ""

admin.site.register(Categoria, CategoriasAdmin)
admin.site.register(Marca, MarcasAdmin)
admin.site.register(Unidad, UnidadesAdmin)
admin.site.register(Producto, ProductoAdmin)
admin.site.register(ProductoCompra, ProductoCompraAdmin)