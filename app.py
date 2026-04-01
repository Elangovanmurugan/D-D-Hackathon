from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from extractor import process_upload

st.set_page_config(page_title='Poster Metadata Extractor', page_icon='🖼️', layout='wide')

MAIN_COLUMNS = [
    'Title', 'Subjects', 'Description', 'Date of Subject', 'Period Start', 'Period End',
    'Creator', 'Collection Name', 'Date Created', 'Lengthmm', 'Breadthmm', 'Orientation',
    'Confidence Score'
]

AUDIT_COLUMNS = [
    'Source File', 'OCR Variant', 'OCR PSM', 'OCR Mean Confidence', 'Raw OCR Text'
]

st.title('🖼️ Theatre Poster Metadata Extractor')
st.write(
    'Upload one or more theatre poster images. The app runs preprocessing, OCR, metadata extraction, '
    'and returns a structured table with confidence scores.'
)

with st.sidebar:
    st.header('How it works')
    st.markdown(
        '1. Upload poster images\\n'
        '2. The app preprocesses each image\\n'
        '3. OCR runs on selected variants\\n'
        '4. Metadata fields are extracted\\n'
        '5. Results can be downloaded as CSV or Excel'
    )

uploaded_files = st.file_uploader(
    'Upload poster images',
    type=['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp', 'webp'],
    accept_multiple_files=True,
)

if uploaded_files:
    records = []
    progress = st.progress(0, text='Starting extraction...')

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        progress.progress(
            (idx - 1) / len(uploaded_files),
            text=f'Processing {uploaded_file.name} ({idx}/{len(uploaded_files)})',
        )
        try:
            records.append(process_upload(uploaded_file, file_name=uploaded_file.name))
        except Exception as exc:  # pragma: no cover - app surface
            st.error(f'Failed to process {uploaded_file.name}: {exc}')

    progress.progress(1.0, text='Extraction complete.')

    if records:
        df = pd.DataFrame(records)
        df_main = df[MAIN_COLUMNS].copy()
        df_audit = df[MAIN_COLUMNS + AUDIT_COLUMNS].copy()

        st.subheader('Metadata table')
        st.dataframe(df_main, use_container_width=True)

        st.subheader('Audit view')
        with st.expander('Show OCR audit fields'):
            st.dataframe(df_audit, use_container_width=True)

        csv_data = df_main.to_csv(index=False).encode('utf-8')
        st.download_button(
            'Download CSV',
            data=csv_data,
            file_name='poster_metadata.csv',
            mime='text/csv',
        )

        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_main.to_excel(writer, sheet_name='metadata', index=False)
            df_audit.to_excel(writer, sheet_name='audit', index=False)
        excel_buffer.seek(0)

        st.download_button(
            'Download Excel',
            data=excel_buffer.getvalue(),
            file_name='poster_metadata.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
else:
    st.info('Upload poster images to begin.')
