# Generated by Django 2.2.6 on 2020-01-16 17:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ami_api', '0006_auto_20200115_2152'),
    ]

    operations = [
        migrations.AddField(
            model_name='overlayimage',
            name='metadatafilepath',
            field=models.URLField(default=''),
        ),
        migrations.AddField(
            model_name='overlayimage',
            name='scalefilepath',
            field=models.URLField(default=''),
        ),
        migrations.AddField(
            model_name='stackedimage',
            name='demfilepath',
            field=models.URLField(default=''),
        ),
        migrations.AlterField(
            model_name='overlayimage',
            name='date',
            field=models.DateField(auto_now=True),
        ),
        migrations.AlterField(
            model_name='overlayimage',
            name='filepath',
            field=models.URLField(default=''),
        ),
        migrations.AlterField(
            model_name='stackedimage',
            name='date',
            field=models.DateField(auto_now=True),
        ),
        migrations.AlterField(
            model_name='stackedimage',
            name='filepath',
            field=models.URLField(default=''),
        ),
    ]
