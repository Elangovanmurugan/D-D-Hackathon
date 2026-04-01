from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = image.convert('RGB')
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def load_image_from_upload(file_obj: BytesIO | bytes | Any) -> Image.Image:
    if hasattr(file_obj, 'read'):
        data = file_obj.read()
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
    elif isinstance(file_obj, bytes):
        data = file_obj
    else:
        raise TypeError('Unsupported file object type')
    return Image.open(BytesIO(data)).convert('RGB')


def load_image_from_path(path: str | Path) -> Image.Image:
    return Image.open(path).convert('RGB')


def normalize_text(txt: str) -> str:
    txt = txt.replace('\x0c', ' ')
    txt = txt.replace('—', '-').replace('–', '-')
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()


def resize_for_speed(img_bgr: np.ndarray, max_width: int = 1400) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w <= max_width:
        return img_bgr
    scale = max_width / w
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def estimate_skew_angle(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) > 10:
        return 0.0
    return float(angle)


def deskew(img_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.2:
        return img_bgr, angle
    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img_bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def preprocess_variants(img_bgr: np.ndarray) -> Dict[str, np.ndarray | float]:
    img_bgr = resize_for_speed(img_bgr)
    img_bgr, angle = deskew(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return {
        'gray': gray,
        'otsu': otsu,
        'deskew_angle': angle,
    }


def ocr_text(image_array: np.ndarray, psm: int = 6) -> str:
    return pytesseract.image_to_string(image_array, config=f'--oem 3 --psm {psm}')


def ocr_df(image_array: np.ndarray, psm: int = 6) -> pd.DataFrame:
    return pytesseract.image_to_data(
        image_array,
        config=f'--oem 3 --psm {psm}',
        output_type=pytesseract.Output.DATAFRAME,
    )


def smart_ocr(variants: Dict[str, np.ndarray | float]) -> Tuple[str, float, str, int]:
    """Fast first-pass OCR with confidence-based fallback."""
    best_text = ''
    best_conf = -1.0
    best_variant = 'gray'
    best_psm = 6

    for variant_name in ('gray', 'otsu'):
        arr = variants[variant_name]
        try:
            text = normalize_text(ocr_text(arr, psm=6))
            df = ocr_df(arr, psm=6)
            if df is None or df.empty or 'conf' not in df.columns:
                mean_conf = 0.0
            else:
                tmp = df[df['text'].notna()].copy()
                tmp['text'] = tmp['text'].astype(str)
                tmp = tmp[tmp['text'].str.strip() != '']
                tmp['conf'] = pd.to_numeric(tmp['conf'], errors='coerce')
                mean_conf = float(tmp['conf'].fillna(0).clip(lower=0).mean()) if not tmp.empty else 0.0
        except Exception:
            continue

        score = mean_conf
        if len(text) < 80:
            score -= 20

        if score > best_conf:
            best_text = text
            best_conf = mean_conf
            best_variant = variant_name
            best_psm = 6

    return best_text, max(best_conf, 0.0), best_variant, best_psm


def extract_date_subject(text: str) -> Optional[str]:
    patterns = [
        re.compile(r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+([A-Z][a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})', re.IGNORECASE),
        re.compile(r'([A-Z][a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})', re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(text)
        if not m:
            continue
        if len(m.groups()) == 4:
            _, month, day, year = m.groups()
        else:
            month, day, year = m.groups()
        return f"{month.title()[:3]} {int(day)}, {year}"
    return None


def extract_year(text: str) -> Optional[str]:
    years = re.findall(r'\b(18\d{2}|19\d{2}|20\d{2})\b', text)
    if not years:
        return None
    # For these posters, earliest prominent year is usually the poster year.
    return years[0]


def detect_subject_terms(text: str) -> str:
    text_u = text.upper()
    subjects: List[str] = []
    if 'THEATRE' in text_u or 'VARIETIES' in text_u:
        subjects.append('Theatre')
    if 'VARIETY' in text_u or 'VARIETIES' in text_u:
        subjects.append('Variety Show')
    if any(k in text_u for k in ['COMEDY', 'COMEDIAN', 'LAUGH']):
        subjects.append('Comedy')
    if any(k in text_u for k in ['VOCAL', 'SONG', 'SINGER', 'QUINTETTE', 'ORCHESTRA', 'MUSIC']):
        subjects.append('Music Hall')
    if 'BIOSCOPE' in text_u or 'CINEMA' in text_u:
        subjects.append('Early Cinema')
    if any(k in text_u for k in ['DANCE', 'DANCER', 'DANSEUSE']):
        subjects.append('Dance')
    if any(k in text_u for k in ['ATHLETIC', 'JUGGLING', 'ACROBAT', 'NOVELTY', 'SPECIALITY']):
        subjects.append('Speciality Acts')
    seen = set()
    ordered = []
    for item in subjects:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ', '.join(ordered)


def extract_acts(text: str, max_names: int = 8) -> List[str]:
    candidates = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z&\']+){0,3})\b', text)
    blacklist = {
        'Argyll Theatre', 'Theatre Of', 'Theatre Varieties', 'Brown Royal', 'Royal Bioscope',
        'Private Boxes', 'Monday', 'Twice Nightly', 'Doors Open', 'Manager', 'Birkenhead'
    }
    cleaned: List[str] = []
    for cand in candidates:
        cand = cand.strip(" ,.-")
        if len(cand) < 4 or cand.isupper():
            continue
        if cand in blacklist:
            continue
        if any(char.isdigit() for char in cand):
            continue
        if cand not in cleaned:
            cleaned.append(cand)
    return cleaned[:max_names]


def infer_title(path_name: str, text: str) -> str:
    acts = extract_acts(text, max_names=2)
    if len(acts) >= 2:
        return f'Argyll Theatre Poster – {acts[0]} & {acts[1]}'
    if len(acts) == 1:
        return f'Argyll Theatre Poster – {acts[0]}'
    stem = Path(path_name).stem
    return f'Argyll Theatre Poster – {stem}'


def generic_description(text: str) -> str:
    subjects = detect_subject_terms(text)
    acts = extract_acts(text, max_names=6)
    if acts:
        acts_text = ', '.join(acts)
        return (
            'Poster advertising a theatre variety bill at the Argyll Theatre of Varieties, '
            f'Birkenhead, featuring acts including {acts_text}. '
            + (f'Subjects inferred from the poster: {subjects}.' if subjects else '')
        ).strip()
    if subjects:
        return (
            'Poster advertising a theatre variety bill at the Argyll Theatre of Varieties, '
            f'Birkenhead. Subjects inferred from the poster: {subjects}.'
        )
    return 'Poster advertising a theatre variety bill at the Argyll Theatre of Varieties, Birkenhead.'


def path_to_orientation(image: Image.Image) -> str:
    w, h = image.size
    if h > w:
        return 'Portrait'
    if w > h:
        return 'Landscape'
    return 'Square'


def dimensions_mm(image: Image.Image) -> Tuple[Optional[int], Optional[int]]:
    dpi = image.info.get('dpi')
    if not dpi or len(dpi) < 2:
        return None, None
    xdpi, ydpi = dpi[0], dpi[1]
    if not xdpi or not ydpi:
        return None, None
    w_px, h_px = image.size
    width_mm = int(round((w_px / xdpi) * 25.4))
    height_mm = int(round((h_px / ydpi) * 25.4))
    return height_mm, width_mm


def overall_confidence(record: Dict[str, Any], ocr_conf: float, text: str) -> str:
    score = 0.0
    score += min(ocr_conf, 100) * 0.55
    if record.get('Date of Subject'):
        score += 15
    if record.get('Period Start'):
        score += 10
    if record.get('Subjects'):
        score += 10
    if record.get('Orientation'):
        score += 5
    if len(text) > 200:
        score += 5
    return f"{int(round(min(score, 99)))}%"


def process_pil_image(image: Image.Image, source_name: str = 'uploaded_image') -> Dict[str, Any]:
    img_bgr = pil_to_bgr(image)
    variants = preprocess_variants(img_bgr)
    text, ocr_conf, variant, psm = smart_ocr(variants)
    text = normalize_text(text)

    date_subject = extract_date_subject(text)
    year = extract_year(text)
    length_mm, breadth_mm = dimensions_mm(image)

    record: Dict[str, Any] = {
        'Title': infer_title(source_name, text),
        'Subjects': detect_subject_terms(text),
        'Description': generic_description(text),
        'Date of Subject': date_subject,
        'Period Start': year,
        'Period End': year,
        'Creator': 'Unknown',
        'Collection Name': 'Argyll Theatre Posters Collection',
        'Date Created': f'c.{year}' if year else None,
        'Lengthmm': length_mm,
        'Breadthmm': breadth_mm,
        'Orientation': path_to_orientation(image),
        'Confidence Score': overall_confidence({
            'Date of Subject': date_subject,
            'Period Start': year,
            'Subjects': detect_subject_terms(text),
            'Orientation': path_to_orientation(image),
        }, ocr_conf, text),
        'Source File': source_name,
        'OCR Variant': variant,
        'OCR PSM': psm,
        'OCR Mean Confidence': round(ocr_conf, 2),
        'Raw OCR Text': text,
    }

    if record['Lengthmm'] is None:
        record['Lengthmm'] = 'Unknown'
    if record['Breadthmm'] is None:
        record['Breadthmm'] = 'Unknown'

    return record


def process_upload(file_obj: Any, file_name: Optional[str] = None) -> Dict[str, Any]:
    image = load_image_from_upload(file_obj)
    source_name = file_name or getattr(file_obj, 'name', 'uploaded_image')
    return process_pil_image(image, source_name=source_name)


def process_paths(paths: Iterable[str | Path]) -> pd.DataFrame:
    records = []
    for path in paths:
        path = Path(path)
        if path.suffix.lower() not in VALID_EXTS:
            continue
        image = load_image_from_path(path)
        records.append(process_pil_image(image, source_name=path.name))
    return pd.DataFrame(records)
