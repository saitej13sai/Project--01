"""
PDF & Text Resume Extraction Module
Supports: PDF, TXT, and plain text input
"""

import re
import io

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


def extract_text_from_pdf(file_obj) -> str:
    """Extract text from a PDF file object."""
    if not PDFPLUMBER_AVAILABLE:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")

    text_parts = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return '\n'.join(text_parts)


def extract_text_from_txt(file_obj) -> str:
    """Extract text from a TXT file object."""
    raw = file_obj.read()
    if isinstance(raw, bytes):
        try:
            return raw.decode('utf-8')
        except UnicodeDecodeError:
            return raw.decode('latin-1', errors='replace')
    return raw


def extract_text(file_obj, filename: str) -> str:
    """Auto-detect file type and extract text."""
    fname = filename.lower()

    if fname.endswith('.pdf'):
        return extract_text_from_pdf(file_obj)
    elif fname.endswith('.txt'):
        return extract_text_from_txt(file_obj)
    else:
        # Try as text
        try:
            return extract_text_from_txt(file_obj)
        except Exception:
            raise ValueError(f"Unsupported file format: {filename}")


def clean_extracted_text(text: str) -> str:
    """Clean up extracted text for better analysis."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    return text.strip()


def extract_candidate_name(text: str) -> str:
    """Heuristic: first non-empty line is usually the candidate name."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        first = lines[0]
        # Name lines are short and title-cased
        if len(first.split()) <= 5 and len(first) < 50:
            return first
    return "Unknown Candidate"


def extract_contact_info(text: str) -> dict:
    """Extract email, phone from resume text."""
    email = re.findall(r'[\w.\-+]+@[\w.\-]+\.\w{2,}', text)
    phone = re.findall(r'[\+]?[\d\s\-().]{10,15}', text)
    linkedin = re.findall(r'linkedin\.com/in/[\w\-]+', text, re.IGNORECASE)
    github = re.findall(r'github\.com/[\w\-]+', text, re.IGNORECASE)

    return {
        "email": email[0] if email else None,
        "phone": phone[0].strip() if phone else None,
        "linkedin": linkedin[0] if linkedin else None,
        "github": github[0] if github else None,
    }
