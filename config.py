import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'pixel_revive_secret_key_change_me')

    # Database Configuration
    # Render provides DATABASE_URL; if not set, default to local SQLite
    _db_url = os.environ.get('DATABASE_URL')
    if _db_url:
        if _db_url.startswith('postgres://'):
            _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
        SQLALCHEMY_DATABASE_URI = _db_url
        SQLALCHEMY_ENGINE_OPTIONS = {
            'pool_pre_ping': True,
            'pool_recycle': 280,
            'connect_args': {'connect_timeout': 10}
        }
    else:
        # Local development fallback
        SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pixelrevive.db')
        SQLALCHEMY_ENGINE_OPTIONS = {
            'pool_pre_ping': True
        }
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload settings
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max limit
