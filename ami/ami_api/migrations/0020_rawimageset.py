# Generated by Django 2.2.6 on 2020-01-23 04:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ami_api', '0019_auto_20200121_1137'),
    ]

    operations = [
        migrations.CreateModel(
            name='RawImageSet',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False, unique=True)),
                ('user', models.CharField(max_length=60)),
                ('field', models.CharField(max_length=60)),
                ('date', models.DateField()),
                ('filepath', models.CharField(max_length=200)),
            ],
        ),
    ]
