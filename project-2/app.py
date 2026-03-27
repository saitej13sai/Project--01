"""
AI-Powered Resume Ranker — Flask Application
"""

import os
import json
import uuid
import io
from datetime import datetime
from flask import (
    Flask, request, jsonify, render_template,
    send_file, session, make_response
)
from werkzeug.utils import secure_filename

from scorer import ResumeScorer
from extractor import extract_text, clean_extracted_text, extract_candidate_name, extract_contact_info
from report_generator import generate_csv_report, generate_summary_stats

# ─── Config ───────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'txt'}
MAX_RESUMES = 20
MAX_FILE_SIZE_MB = 5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory session store (use Redis for production)
SESSION_STORE = {}

scorer = ResumeScorer()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/rank', methods=['POST'])
def rank_resumes():
    """Main ranking endpoint."""
    try:
        # Get job description
        job_description = request.form.get('job_description', '').strip()
        job_title = request.form.get('job_title', 'Position').strip()
        job_skills_raw = request.form.get('job_skills', '').strip()

        if not job_description:
            return jsonify({"error": "Job description is required."}), 400

        if len(job_description) < 50:
            return jsonify({"error": "Job description is too short (min 50 chars)."}), 400

        # Parse manual skills
        job_skills = [s.strip() for s in job_skills_raw.split(',') if s.strip()]

        # Get uploaded files
        files = request.files.getlist('resumes')
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "Please upload at least one resume."}), 400

        if len(files) > MAX_RESUMES:
            return jsonify({"error": f"Maximum {MAX_RESUMES} resumes allowed."}), 400

        # Extract text from each resume
        resumes = []
        errors = []

        for file in files:
            if not file or file.filename == '':
                continue
            if not allowed_file(file.filename):
                errors.append(f"{file.filename}: Unsupported format (PDF/TXT only)")
                continue

            try:
                filename = secure_filename(file.filename)
                file_bytes = file.read()
                file_size = len(file_bytes)

                if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    errors.append(f"{filename}: File too large (max {MAX_FILE_SIZE_MB}MB)")
                    continue

                file_obj = io.BytesIO(file_bytes)
                raw_text = extract_text(file_obj, filename)
                clean_text = clean_extracted_text(raw_text)

                if len(clean_text) < 100:
                    errors.append(f"{filename}: Could not extract sufficient text")
                    continue

                candidate_name = extract_candidate_name(clean_text)
                contact_info = extract_contact_info(clean_text)

                resumes.append({
                    "filename": filename,
                    "candidate_name": candidate_name,
                    "text": clean_text,
                    "size": file_size,
                    "contact": contact_info,
                })

            except Exception as e:
                errors.append(f"{file.filename}: {str(e)}")

        if not resumes:
            return jsonify({"error": "No valid resumes could be processed.", "file_errors": errors}), 400

        # Score and rank
        raw_results = scorer.rank_resumes(resumes, job_description, job_skills)

        # Enrich with contact info and candidate name
        for result in raw_results:
            fname = result['filename']
            original = next((r for r in resumes if r['filename'] == fname), {})
            result['candidate_name'] = original.get('candidate_name', fname)
            result['contact'] = original.get('contact', {})

        # Generate stats
        stats = generate_summary_stats(raw_results)

        # Store results in session
        sid = get_session_id()
        SESSION_STORE[sid] = {
            "results": raw_results,
            "job_title": job_title,
            "job_description": job_description,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
        }

        return jsonify({
            "success": True,
            "results": raw_results,
            "stats": stats,
            "job_title": job_title,
            "file_errors": errors,
            "processed_count": len(resumes),
        })

    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


@app.route('/api/download-report', methods=['GET'])
def download_report():
    """Download CSV report for the last ranking session."""
    try:
        sid = get_session_id()
        session_data = SESSION_STORE.get(sid)

        if not session_data:
            return jsonify({"error": "No ranking results found. Please run a ranking first."}), 404

        results = session_data['results']
        job_title = session_data['job_title']

        csv_content = generate_csv_report(results, job_title)

        response = make_response(csv_content)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"resume_ranking_{job_title.replace(' ', '_')}_{timestamp}.csv"
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        response.headers['Content-Type'] = 'text/csv'

        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  AI Resume Ranker — Starting Server")
    print("  Open: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
