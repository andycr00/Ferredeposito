from django.urls import path
from . import views

urlpatterns = [
    path('ventas/', views.Prueba.as_view()),
    path('lista/', views.PruebaList.as_view()),
]
