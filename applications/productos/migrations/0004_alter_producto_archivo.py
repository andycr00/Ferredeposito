# Generated by Django 4.0.5 on 2022-06-23 17:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('productos', '0003_alter_producto_lista_precios'),
    ]

    operations = [
        migrations.AlterField(
            model_name='producto',
            name='archivo',
            field=models.FileField(blank=True, null=True, upload_to='media'),
        ),
    ]