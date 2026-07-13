from app import app
from models import db
from sqlalchemy import text

def initialize_database():
    with app.app_context():
        print("Creating database tables for PixelRevive AI...")
        db.create_all()
        
        # Run migration query to ensure columns exist in case the table was created before
        print("Running table migration checks...")
        try:
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE restoration_images ADD COLUMN IF NOT EXISTS processed_filename VARCHAR(255);"))
                conn.execute(text("ALTER TABLE restoration_images ADD COLUMN IF NOT EXISTS duration DOUBLE PRECISION;"))
                conn.execute(text("ALTER TABLE restoration_images ADD COLUMN IF NOT EXISTS faces_detected INTEGER DEFAULT 0;"))
                conn.execute(text("ALTER TABLE restoration_images ADD COLUMN IF NOT EXISTS original_resolution VARCHAR(50);"))
                conn.execute(text("ALTER TABLE restoration_images ADD COLUMN IF NOT EXISTS output_resolution VARCHAR(50);"))
                conn.execute(text("ALTER TABLE restoration_images ADD COLUMN IF NOT EXISTS file_size INTEGER;"))
                conn.execute(text("ALTER TABLE restoration_images ADD COLUMN IF NOT EXISTS output_file_size INTEGER;"))
                conn.execute(text("ALTER TABLE restoration_images ADD COLUMN IF NOT EXISTS crease_px INTEGER DEFAULT 0;"))
                conn.execute(text("ALTER TABLE restoration_images ADD COLUMN IF NOT EXISTS scratch_px INTEGER DEFAULT 0;"))
                conn.execute(text("ALTER TABLE restoration_images ADD COLUMN IF NOT EXISTS total_damage_pct DOUBLE PRECISION DEFAULT 0.0;"))
                conn.execute(text("ALTER TABLE restoration_images ADD COLUMN IF NOT EXISTS mask_coverage_pct DOUBLE PRECISION DEFAULT 0.0;"))
                conn.commit()
            print("Migration checks complete!")
        except Exception as e:
            print(f"Migration error: {e}")

if __name__ == '__main__':
    initialize_database()
