import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'pixel_revive_secret_key_change_me')

    # Database Configuration
    # Render provides DATABASE_URL with legacy "postgres://" prefix;
    # SQLAlchemy requires "postgresql://", so we fix it here.
    _db_url = os.environ.get(
        'DATABASE_URL',
        'postgresql://pixel_user:pixel_secure_pass@localhost:5432/pixelrevive'
    )
    if _db_url.startswith('postgres://'):
        _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
    SQLALCHEMY_DATABASE_URI = _db_url
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload settings
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max limit
