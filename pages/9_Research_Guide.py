from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.guide import SECTIONS
from src.ui import page_header

page_header(
    "Documentation and Research Notes",
    "File rules, units, index definitions, modelling workflow, storage, and the limits of this prototype.",
)

for section in SECTIONS:
    st.subheader(section["title"])
    for paragraph in section["paragraphs"]:
        st.write(paragraph)
