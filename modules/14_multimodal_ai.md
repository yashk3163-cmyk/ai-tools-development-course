# Module 14: Multi-Modal AI — Vision, Audio, and Beyond

> **Estimated Reading Time:** 3 hours  
> **Outcome:** Use GPT-4o for image and document analysis, Whisper for audio transcription, and build multi-modal AI workflows.

---

## 14.1 Why Multi-Modal AI Matters for Professional Practice

A significant portion of professional work involves non-text data: scanned documents with handwriting, photographs of site conditions, audio recordings of client meetings, financial charts and balance sheets, photographs of evidence. Until recently, AI could only process text. Multi-modal AI extends the AI's perceptual world.

For a CA or advocate in India, multi-modal AI enables:
- Extracting text from scanned legal documents and orders
- Reading handwritten notes and converting to structured data
- Transcribing client meeting recordings to searchable text
- Analyzing financial statement images and charts
- Processing photographed receipts and invoices

---

```python
from openai import OpenAI
from dotenv import load_dotenv
import base64, json, os
from pathlib import Path

load_dotenv()
client = OpenAI()

# ===== Vision: Analyze Images =====
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def analyze_document_image(image_path: str, task: str = "Extract all text and data from this document.") -> str:
    """Analyze a scanned document or image."""
    img_base64 = encode_image(image_path)
    ext = Path(image_path).suffix.lower()
    media_type = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                  '.png': 'image/png', '.pdf': 'application/pdf'}.get(ext, 'image/jpeg')
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": task},
                {"type": "image_url",
                 "image_url": {"url": f"data:{media_type};base64,{img_base64}",
                               "detail": "high"}}
            ]
        }],
        max_tokens=2000
    )
    return response.choices[0].message.content

def extract_invoice_data(image_path: str) -> dict:
    """Extract structured data from invoice/receipt image."""
    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": """
                Extract invoice data as JSON:
                {vendor_name, invoice_number, date, gstin_vendor, gstin_buyer,
                 line_items: [{description, quantity, rate, amount}],
                 subtotal, gst_amount, gst_rate, total_amount, payment_terms}
                """},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
            ]
        }],
        response_format={"type": "json_object"},
        max_tokens=1000
    )
    return json.loads(result.choices[0].message.content)

# ===== Audio: Transcription with Whisper =====
def transcribe_audio(audio_file_path: str, language: str = "en") -> dict:
    """Transcribe audio file using OpenAI Whisper."""
    with open(audio_file_path, 'rb') as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return {
        "text": transcript.text,
        "language": transcript.language,
        "duration": transcript.duration,
        "segments": [{"start": s.start, "end": s.end, "text": s.text}
                     for s in transcript.segments]
    }

def transcribe_client_meeting(audio_path: str) -> dict:
    """Transcribe a client meeting and extract action items."""
    transcript_data = transcribe_audio(audio_path)
    
    analysis = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "system",
            "content": "You are an expert CA and legal professional. Analyze meeting transcripts."
        }, {
            "role": "user",
            "content": f"""
            Meeting transcript:\n{transcript_data['text']}\n\n
            Extract as JSON:
            - meeting_summary (2-3 sentences)
            - client_concerns (list)
            - tax_issues_discussed (list with section references)
            - action_items (list with responsible party and deadline if mentioned)
            - follow_up_required (bool)
            """
        }],
        response_format={"type": "json_object"}
    )
    
    return {
        "transcript": transcript_data,
        "analysis": json.loads(analysis.choices[0].message.content)
    }

# ===== Multi-Modal: URL-based (no local file needed) =====
def analyze_image_url(url: str, task: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": task},
                {"type": "image_url", "image_url": {"url": url}}
            ]
        }],
        max_tokens=1500
    )
    return response.choices[0].message.content

# Example: Analyze a company's balance sheet image
result = analyze_image_url(
    "https://example.com/balance_sheet.jpg",
    """You are a CA. Analyze this balance sheet:
    1. Extract key financial ratios (current ratio, debt-equity ratio)
    2. Identify any red flags
    3. Calculate working capital
    4. Comment on financial health"""
)
```
