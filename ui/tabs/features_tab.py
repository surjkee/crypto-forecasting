# ui/tabs/features_tab.py

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import get_settings
from features.transform import build_feature_frame
from ui.constants import TRACKED_COINS
from ui.tabs.data_tab import load_market_data


@st.cache_data(show_spinner=False)
def compute_features(
    coin_id: str,
    vs_currency: str,
) -> pd.DataFrame:
    """
    Обчислює технічні фічі поверх історичних даних.
    """
    df = load_market_data(coin_id, vs_currency)
    if df.empty:
        return df

    feat_df = build_feature_frame(df)
    return feat_df


def render_features_tab():
    settings = get_settings()
    vs_currency = settings.default_vs_currency

    st.markdown("""
<style>
        /* Remove blank space at top and bottom */ 
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
</style>
""", unsafe_allow_html=True)

    st.markdown(
    """
    <h1 style="text-align: center; margin-top: 0;">
        🧩 Features
    </h1>
    """,
    unsafe_allow_html=True
)
    # Використовуємо глобально обрану монету з session_state
    labels = [label for label, _ in TRACKED_COINS]
    ids = [cid for _, cid in TRACKED_COINS]

    default_index = ids.index("bitcoin") if "bitcoin" in ids else 0
    default_label = labels[default_index]
    default_id = ids[default_index]

    selected_coin_id = st.session_state.get("selected_coin_id", default_id)
    selected_label = st.session_state.get("selected_coin_label", default_label)

    # Завантажуємо дані + фічі
    df_raw = load_market_data(selected_coin_id, vs_currency)
    if df_raw.empty:
        st.warning(
            "У базі немає даних для цієї монети. "
            "Спочатку запусти job для завантаження історії:\n\n"
            "`python -m jobs.fetch_history`"
        )
        return

    df_feat = compute_features(selected_coin_id, vs_currency)

    # --- 2x2 grid з графіками ---
    st.subheader(f"Metrics for {selected_label}")

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # 1) Ціна + SMA
    with row1_col1:
        st.markdown("**Price + SMA**")
        fig_price = px.line(
            df_feat,
            x="ts",
            y=["price", "sma_5", "sma_20", "sma_50"],
            labels={"value": f"Price / SMA ({vs_currency.upper()})", "ts": "Time"},
        )
        fig_price.update_layout(height=350, legend_title_text="Серія")
        st.plotly_chart(fig_price, width="stretch")

    # 2) Returns
    with row1_col2:
        st.markdown("**Returns**")
        fig_ret = px.line(
            df_feat,
            x="ts",
            y="return",
            labels={"return": "Return", "ts": "Time"},
        )
        fig_ret.update_layout(height=350)
        st.plotly_chart(fig_ret, width="stretch")

    # 3) Волатильність
    vol_col = "volatility_24"
    with row2_col1:
        st.markdown("**Volatility (24 points)**")
        if vol_col in df_feat.columns:
            fig_vol = px.line(
                df_feat,
                x="ts",
                y=vol_col,
                labels={vol_col: "Volatility (σ)", "ts": "Time"},
            )
            fig_vol.update_layout(height=350)
            st.plotly_chart(fig_vol, width="stretch")
        else:
            st.info("Volatility not calculated yet.")

    # 4) Обʼєм
    with row2_col2:
        st.markdown("**Volume**")
        fig_volm = px.line(
            df_feat,
            x="ts",
            y="volume",
            labels={"volume": "Volume", "ts": "Time"},
        )
        fig_volm.update_layout(height=350)
        st.plotly_chart(fig_volm, width="stretch")

    with st.expander("Features Table"):
        st.dataframe(
            df_feat.sort_values("ts", ascending=False),
            width="stretch",
            height=400,
        )
