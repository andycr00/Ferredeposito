# Generated by Django 4.0.5 on 2022-06-23 05:50

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('compras', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='CierreCaja',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('fecha', models.DateField(auto_now_add=True, null=True)),
                ('observacion', models.CharField(max_length=300)),
                ('valor', models.FloatField()),
                ('creado_en', models.DateTimeField(auto_now_add=True, null=True)),
                ('actualizado_en', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Cobro',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('descripcion', models.CharField(max_length=300)),
                ('fecha_pago', models.DateTimeField(blank=True, null=True)),
                ('valor', models.FloatField(blank=True, null=True)),
                ('creado_en', models.DateTimeField(auto_now_add=True, null=True)),
                ('actualizado_en', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Deuda',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('saldo', models.FloatField(blank=True, null=True)),
                ('fecha_inicio', models.DateTimeField(auto_now_add=True, null=True)),
                ('fecha_fin', models.DateTimeField(blank=True, null=True)),
                ('estado', models.BooleanField(default=True, null=True)),
                ('creado_en', models.DateTimeField(auto_now_add=True, null=True)),
                ('actualizado_en', models.DateTimeField(auto_now=True, null=True)),
                ('compra', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='compras.compra')),
            ],
        ),
        migrations.CreateModel(
            name='Gasto',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('titulo', models.CharField(blank=True, max_length=200, null=True)),
                ('descripcion', models.CharField(blank=True, max_length=255, null=True)),
                ('valor', models.FloatField(blank=True, null=True)),
                ('fecha', models.DateTimeField(auto_now_add=True, null=True)),
                ('tipo', models.CharField(blank=True, choices=[('PERSONALES', 'Personales'), ('FIJOS', 'Fijos')], default='FIJOS', max_length=200, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Notificacion',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('titulo', models.CharField(max_length=128)),
                ('descripcion', models.CharField(max_length=128)),
                ('leido', models.BooleanField(blank=True, default=False, null=True)),
                ('creado_en', models.DateTimeField(auto_now_add=True, null=True)),
                ('actualizado_en', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Usuario',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nombres', models.CharField(max_length=200)),
                ('apellidos', models.CharField(max_length=200)),
                ('cedula', models.CharField(max_length=20)),
                ('telefono', models.CharField(max_length=50)),
                ('correo', models.CharField(max_length=50)),
                ('direccion', models.CharField(max_length=50)),
                ('creado_en', models.DateTimeField(auto_now_add=True, null=True)),
                ('actualizado_en', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Pago',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('titulo', models.CharField(blank=True, max_length=200, null=True)),
                ('descripcion', models.CharField(blank=True, max_length=200, null=True)),
                ('fecha', models.DateTimeField(blank=True, null=True)),
                ('valor', models.FloatField(blank=True, null=True)),
                ('tipo', models.CharField(blank=True, choices=[('ABONO', 'Abono'), ('CANCELACION', 'Cancelacion')], default='ABONO', max_length=200, null=True)),
                ('creado_en', models.DateTimeField(auto_now_add=True, null=True)),
                ('actualizado_en', models.DateTimeField(auto_now=True, null=True)),
                ('deuda', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='administracion.deuda')),
            ],
        ),
    ]
