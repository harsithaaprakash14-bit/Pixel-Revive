import os
import uuid
import time
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
from config import Config
from models import db, RestorationImage
from services.ai_connector import process_image
from services.damage_remover import generate_damage_mask

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db.init_app(app)

# Auto-create database tables on first startup (idempotent)
with app.app_context():
    db.create_all()

ALLOWED_EXTENSIONS = app.config['ALLOWED_EXTENSIONS']

# Ensure required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """
    Check if the file extension is allowed.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """
    Render the main application page.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload, validation, and trigger the AI processing pipeline (or only analysis).
    """
    # 1. Check if the request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['image']
    
    # 2. Check if the user submitted an empty file selection
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # 3. Validate file type and process
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)
        
        # Get file size & resolution
        file_size = os.path.getsize(input_path)
        try:
            with Image.open(input_path) as img:
                w, h = img.size
                original_resolution = f"{w}x{h}"
        except Exception:
            w, h = 800, 600
            original_resolution = "Unknown"
        
        # Save record to database (initial placeholder values, updated during processing)
        try:
            image_record = RestorationImage(
                filename=unique_filename,
                file_path=input_path,
                status='uploaded',
                original_resolution=original_resolution,
                file_size=file_size,
                crease_px=0,
                scratch_px=0,
                faces_detected=0,
                total_damage_pct=0.0,
                mask_coverage_pct=0.0
            )
            db.session.add(image_record)
            db.session.commit()
        except Exception as db_err:
            db.session.rollback()
            return jsonify({'error': f'Database logging failed: {str(db_err)}'}), 500
        
        # 4. Trigger the AI processing pipeline
        image_record.status = 'processing'
        db.session.commit()
        
        start_time = time.time()
        try:
            result = process_image(input_path)
            duration = time.time() - start_time
            processed_path = result['processed_path']
            processed_filename = os.path.basename(processed_path)
            faces_detected = result['faces_detected']
            restoration_meta = result.get('restoration_meta', {})
            
            # Fetch output specs
            output_resolution = "Unknown"
            output_file_size = None
            try:
                with Image.open(processed_path) as out_img:
                    out_w, out_h = out_img.size
                    output_resolution = f"{out_w}x{out_h}"
                output_file_size = os.path.getsize(processed_path)
            except Exception:
                pass
            
            # Update status to restored in database
            image_record.status = 'restored'
            image_record.processed_filename = processed_filename
            image_record.duration = duration
            image_record.faces_detected = faces_detected
            image_record.output_resolution = output_resolution
            image_record.output_file_size = output_file_size
            
            if restoration_meta:
                image_record.crease_px = restoration_meta.get('crease_px', image_record.crease_px)
                image_record.scratch_px = restoration_meta.get('scratch_px', image_record.scratch_px)
                image_record.total_damage_pct = restoration_meta.get('total_damage_pct', image_record.total_damage_pct)
                image_record.mask_coverage_pct = restoration_meta.get('mask_coverage_pct', image_record.mask_coverage_pct)
                
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            duration = time.time() - start_time
            print(f"[API ERROR] Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()

            try:
                image_record.status = 'failed'
                image_record.duration = duration
                db.session.commit()
            except Exception as db_fail_err:
                print(f"[API ERROR] Failed to update fail status in DB: {db_fail_err}")
                db.session.rollback()

            return jsonify({'error': str(e)}), 500
        
        return jsonify({
            'status': 'success',
            'message': 'Image uploaded successfully!',
            'original_image': unique_filename,
            'processed_image': processed_filename,
            'faces_detected': faces_detected,
            'duration': round(duration, 2)
        }), 200
        
    return jsonify({'error': 'File type not allowed. Only JPG, JPEG, and PNG are permitted.'}), 400


@app.route('/restore/<int:image_id>', methods=['POST'])
def restore_image_by_id(image_id):
    """
    Run the full AI pipeline on an already uploaded/analyzed image.
    """
    image_record = RestorationImage.query.get_or_404(image_id)
    
    # If already restored, return the cached details
    if image_record.status == 'restored' and image_record.processed_filename:
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], image_record.processed_filename)
        if os.path.exists(out_path):
            return jsonify({
                'status': 'success',
                'message': 'Image already restored (cached result)',
                'original_image': image_record.filename,
                'processed_image': image_record.processed_filename,
                'faces_detected': image_record.faces_detected,
                'duration': round(image_record.duration or 0.0, 2)
            }), 200

    image_record.status = 'processing'
    db.session.commit()
    
    start_time = time.time()
    try:
        result = process_image(image_record.file_path)
        duration = time.time() - start_time
        processed_path = result['processed_path']
        processed_filename = os.path.basename(processed_path)
        faces_detected = result['faces_detected']
        restoration_meta = result.get('restoration_meta', {})
        
        # Save output specs
        output_resolution = "Unknown"
        output_file_size = None
        try:
            with Image.open(processed_path) as out_img:
                out_w, out_h = out_img.size
                output_resolution = f"{out_w}x{out_h}"
            output_file_size = os.path.getsize(processed_path)
        except Exception:
            pass
            
        image_record.status = 'restored'
        image_record.processed_filename = processed_filename
        image_record.duration = duration
        image_record.faces_detected = faces_detected
        image_record.output_resolution = output_resolution
        image_record.output_file_size = output_file_size
        
        if restoration_meta:
            image_record.crease_px = restoration_meta.get('crease_px', image_record.crease_px)
            image_record.scratch_px = restoration_meta.get('scratch_px', image_record.scratch_px)
            image_record.total_damage_pct = restoration_meta.get('total_damage_pct', image_record.total_damage_pct)
            image_record.mask_coverage_pct = restoration_meta.get('mask_coverage_pct', image_record.mask_coverage_pct)
            
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        duration = time.time() - start_time
        print(f"[API ERROR] Restore ID {image_id} failed: {str(e)}")
        
        try:
            image_record.status = 'failed'
            image_record.duration = duration
            db.session.commit()
        except Exception as db_fail_err:
            print(f"[API ERROR] Failed to update fail status in DB: {db_fail_err}")
            db.session.rollback()
            
        return jsonify({'error': str(e)}), 500
        
    return jsonify({
        'status': 'success',
        'message': 'Image restored successfully!',
        'original_image': image_record.filename,
        'processed_image': processed_filename,
        'faces_detected': faces_detected,
        'duration': round(duration, 2)
    }), 200


@app.route('/api/image/<int:image_id>', methods=['GET'])
def get_image_details(image_id):
    """
    Return JSON details of a single restoration history entry.
    """
    image_record = RestorationImage.query.get_or_404(image_id)
    return jsonify(image_record.to_dict()), 200

@app.route('/history')
def history():
    """
    Render the restoration history page with search, sort, and filter capabilities.
    """
    query = RestorationImage.query
    
    # Filter by filename
    search = request.args.get('search', '')
    if search:
        query = query.filter(RestorationImage.filename.ilike(f'%{search}%'))
        
    # Filter by status
    status = request.args.get('status', '')
    if status:
        query = query.filter(RestorationImage.status == status)
        
    # Filter by date range (direct inputs)
    start_date = request.args.get('start_date', '')
    if start_date:
        query = query.filter(RestorationImage.upload_time >= start_date)
    end_date = request.args.get('end_date', '')
    if end_date:
        query = query.filter(RestorationImage.upload_time <= end_date)
        
    # Predefined date filters
    date_filter = request.args.get('date_filter', '')
    if date_filter == 'today':
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        query = query.filter(RestorationImage.upload_time >= today_start)
    elif date_filter == '7_days':
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        query = query.filter(RestorationImage.upload_time >= seven_days_ago)
    elif date_filter == '30_days':
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        query = query.filter(RestorationImage.upload_time >= thirty_days_ago)
        
    # Sorting
    sort_by = request.args.get('sort_by', 'newest')
    if sort_by == 'oldest':
        query = query.order_by(RestorationImage.upload_time.asc())
    elif sort_by == 'duration':
        query = query.order_by(RestorationImage.duration.asc().nulls_last())
    else:  # newest
        query = query.order_by(RestorationImage.upload_time.desc())
        
    images = query.all()
    
    return render_template('history.html', images=images, filters={
        'search': search,
        'status': status,
        'start_date': start_date,
        'end_date': end_date,
        'date_filter': date_filter,
        'sort_by': sort_by
    })

@app.route('/delete/<int:image_id>', methods=['POST', 'DELETE'])
def delete_image(image_id):
    """
    Delete a history entry and its associated physical files.
    """
    image_record = RestorationImage.query.get_or_404(image_id)
    try:
        # Delete original file
        if os.path.exists(image_record.file_path):
            os.remove(image_record.file_path)
            
        # Delete processed file
        if image_record.processed_filename:
            out_path = os.path.join(app.config['OUTPUT_FOLDER'], image_record.processed_filename)
            if os.path.exists(out_path):
                os.remove(out_path)
                
        db.session.delete(image_record)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Image deleted successfully!'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to delete: {str(e)}'}), 500

@app.route('/admin')
def admin_dashboard():
    """
    Render the admin statistics dashboard.
    """
    total_count = RestorationImage.query.count()
    restored_count = RestorationImage.query.filter_by(status='restored').count()
    failed_count = RestorationImage.query.filter_by(status='failed').count()
    processing_count = RestorationImage.query.filter_by(status='processing').count()
    
    # Calculate averages & aggregates
    avg_duration = db.session.query(db.func.avg(RestorationImage.duration)).filter(RestorationImage.status == 'restored').scalar() or 0.0
    total_faces = db.session.query(db.func.sum(RestorationImage.faces_detected)).scalar() or 0
    success_rate = (restored_count / total_count * 100) if total_count > 0 else 0
    
    recent_images = RestorationImage.query.order_by(RestorationImage.upload_time.desc()).limit(10).all()
    
    return render_template('admin.html', stats={
        'total': total_count,
        'restored': restored_count,
        'failed': failed_count,
        'processing': processing_count,
        'avg_duration': round(avg_duration, 2),
        'total_faces': int(total_faces),
        'success_rate': round(success_rate, 1)
    }, recent_images=recent_images)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded/processed files securely from the correct directory.
    """
    filename = secure_filename(filename)
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(out_path):
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    """
    Serve output images as file downloads.
    """
    filename = secure_filename(filename)
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(out_path):
        return send_from_directory(
            app.config['OUTPUT_FOLDER'],
            filename,
            as_attachment=True,
            download_name=f"pixelrevive_{filename}"
        )
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename,
        as_attachment=True,
        download_name=f"pixelrevive_{filename}"
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
