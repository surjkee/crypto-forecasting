import base64
import streamlit as st
from pathlib import Path


def load_svg_base64(path: str) -> str:
    """Завантажити локальний SVG і повернути base64 data-uri."""
    svg_path = Path(path)
    if not svg_path.exists():
        raise FileNotFoundError(f"SVG file not found: {path}")

    data = svg_path.read_text(encoding="utf-8")
    return base64.b64encode(data.encode("utf-8")).decode()


def render_footer(
    svg_path: str = "ui/assets/coingecko-dark.svg",
    text: str = "Data powered by",
    height: int = 32,
):
    """Рендер футера з фіксованим положенням і SVG логотипом."""
    svg_base64 = load_svg_base64(svg_path)

    st.markdown(
        f"""
<style>
/* Додатковий відступ знизу, щоб контент не "влізав" під footer */
[data-testid="stMain"] {{
    padding-bottom: 3.5rem;
}}

.crypto-footer {{
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    padding: 0.4rem 1.4rem;

    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 0.55rem;

    background-color: rgba(255, 255, 255, 0.0);

   /* background: rgba(10, 12, 20, 0.15);
    backdrop-filter: blur(12px);
    border-top: 1px solid rgba(255, 255, 255, 0.08);*/

    font-size: 1.5rem;
    color: rgba(255,255,255,0.75);

    z-index: 999;
}}
</style>

<div class="crypto-footer">
    <span>{text}</span>
    <img src="data:image/svg+xml;base64,{svg_base64}" height="{height}">
</div>
""",
        unsafe_allow_html=True,
    )
