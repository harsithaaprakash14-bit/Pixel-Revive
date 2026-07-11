import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'pixel_revive_secret_key_change_me')
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'postgresql://pixel_user:pixel_secure_pass@localhost:5432/pixelrevive'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload settings
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max limit
