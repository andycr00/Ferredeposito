from .base import * 
import os
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Database
# https://docs.djangoproject.com/en/4.0/ref/settings/#databases

DATABASES = {
    # 'default': {
    #     'ENGINE': 'django.db.backends.sqlite3',
    #     'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    # }

    #  "default": {
    #     "ENGINE": "mssql",
    #     "NAME": "Ferredeposito",
    #     "USER": "ferre_admin",
    #     "PASSWORD": "WL27yeJ2ggaxkws6",
    #     "HOST": "ANDRES\SQLEXPRESS",
    #     "OPTIONS": {"driver": "ODBC Driver 17 for SQL Server", 
    #     },
    # },

    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": "ferredeposito",
        "USER": "andres",
        "PASSWORD": "This.way.up89",
        "HOST": "localhost",
        "PORT": "5432",
    },
}

STATICFILES_DIRS=[os.path.join(BASE_DIR, "static")]
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media_cdn')

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.0/howto/static-files/

STATIC_URL = 'static/'
