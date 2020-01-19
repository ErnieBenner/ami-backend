# Generated by Django 2.2.6 on 2020-01-16 17:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ami_api', '0013_auto_20200116_1227'),
    ]

    operations = [
        migrations.AlterField(
            model_name='overlayimage',
            name='filepath',
            field=models.FilePathField(default='/../images/default.junk', max_length=200),
        ),
        migrations.AlterField(
            model_name='overlayimage',
            name='metadatafilepath',
            field=models.FilePathField(default='/../images/default.junk', max_length=200),
        ),
        migrations.AlterField(
            model_name='overlayimage',
            name='scalefilepath',
            field=models.FilePathField(default='/../images/default.junk', max_length=200),
        ),
        migrations.AlterField(
            model_name='stackedimage',
            name='demfilepath',
            field=models.FilePathField(default='/../images/default.junk', max_length=200),
        ),
        migrations.AlterField(
            model_name='stackedimage',
            name='filepath',
            field=models.FilePathField(default='/../images/default.junk', max_length=200),
        ),
    ]