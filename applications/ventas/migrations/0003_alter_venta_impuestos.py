# Generated by Django 4.0.5 on 2022-06-23 23:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ventas', '0002_cobro'),
    ]

    operations = [
        migrations.AlterField(
            model_name='venta',
            name='impuestos',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
