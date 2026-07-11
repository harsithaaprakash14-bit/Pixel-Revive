from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

db = SQLAlchemy()

class RestorationImage(db.Model):
    __tablename__ = 'restoration_images'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    processed_filename = db.Column(db.String(255), nullable=True)
    duration = db.Column(db.Float, nullable=True)
    faces_detected = db.Column(db.Integer, default=0, nullable=True)
    upload_time = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    status = db.Column(db.String(50), default='uploaded', nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'file_path': self.file_path,
            'processed_filename': self.processed_filename,
            'duration': self.duration,
            'faces_detected': self.faces_detected,
            'upload_time': self.upload_time.isoformat(),
            'status': self.status
        }
