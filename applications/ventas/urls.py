from django.urls import path
from . import views

urlpatterns = [
    path('ventas/', views.Prueba.as_view()),
    path('update/<slug:pk>', views.ClienteUpdateView.as_view()),
    path('prod/', views.ProdVentaList.as_view()),
]
