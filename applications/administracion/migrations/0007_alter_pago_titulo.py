# Generated by Django 4.0.5 on 2022-11-03 01:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('administracion', '0006_remove_pago_tipo_alter_deuda_estado'),
    ]

    operations = [
        migrations.AlterField(
            model_name='pago',
            name='titulo',
            field=models.CharField(max_length=200),
        ),
    ]
