# Poster Metadata Extractor

A Streamlit app for extracting structured metadata from theatre poster images using image preprocessing, Tesseract OCR, and rule-based NLP.

## Files

- `app.py` — Streamlit web app
- `extractor.py` — OCR + metadata extraction pipeline
- `requirements.txt` — Python dependencies
- `packages.txt` — system dependencies for deployment

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push these files to a GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repo.
3. Set `app.py` as the main file.
4. Deploy.

## Notes

- The original notebook contained a hardcoded API key. Do not commit secrets.
- This deployment version avoids notebook-only code such as Colab Drive mounting.
