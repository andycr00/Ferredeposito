from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('cluster', views.clusters),
    path('prediccion', views.prediccion),
    path('clasificacion', views.clasificacion),
    path('clasificacion2', views.clasificacion2)
]
