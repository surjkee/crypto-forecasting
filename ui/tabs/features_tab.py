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

    st.title("🧩 Features (технічні показники)")

    # Для узгодженості з вкладкою Data переоберемо монету так само:
    labels = [label for label, _ in TRACKED_COINS]
    ids = [cid for _, cid in TRACKED_COINS]

    default_index = ids.index("bitcoin") if "bitcoin" in ids else 0

    col_sel, col_info = st.columns([2, 3])

    with col_sel:
        selected_label = st.selectbox(
            "Оберіть монету:",
            options=labels,
            index=default_index,
            key="features_coin_select",
        )
        selected_coin_id = ids[labels.index(selected_label)]

    with col_info:
        st.caption(
            "Тут ми дивимося на похідні ознаки (features), "
            "які потім підуть у моделі LSTM / CNN-LSTM / Attention."
        )

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
    st.subheader(f"Графіки для {selected_label}")

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # 1) Ціна + SMA
    with row1_col1:
        st.markdown("**Ціна + SMA**")
        fig_price = px.line(
            df_feat,
            x="ts",
            y=["price", "sma_5", "sma_20", "sma_50"],
            labels={"value": f"Ціна / SMA ({vs_currency.upper()})", "ts": "Час"},
        )
        fig_price.update_layout(height=350, legend_title_text="Серія")
        st.plotly_chart(fig_price, width="stretch")

    # 2) Returns
    with row1_col2:
        st.markdown("**Добові зміни (returns)**")
        fig_ret = px.line(
            df_feat,
            x="ts",
            y="return",
            labels={"return": "Return", "ts": "Час"},
        )
        fig_ret.update_layout(height=350)
        st.plotly_chart(fig_ret, width="stretch")

    # 3) Волатильність
    vol_col = "volatility_24"
    with row2_col1:
        st.markdown("**Ковзна волатильність (24 точки)**")
        if vol_col in df_feat.columns:
            fig_vol = px.line(
                df_feat,
                x="ts",
                y=vol_col,
                labels={vol_col: "Volatility (σ)", "ts": "Час"},
            )
            fig_vol.update_layout(height=350)
            st.plotly_chart(fig_vol, width="stretch")
        else:
            st.info("Колонка волатильності ще не розрахована.")

    # 4) Обʼєм
    with row2_col2:
        st.markdown("**Обʼєм торгів (volume)**")
        fig_volm = px.line(
            df_feat,
            x="ts",
            y="volume",
            labels={"volume": "Обʼєм", "ts": "Час"},
        )
        fig_volm.update_layout(height=350)
        st.plotly_chart(fig_volm, width="stretch")

    with st.expander("Показати таблицю з фічами"):
        st.dataframe(
            df_feat.sort_values("ts", ascending=False),
            width="stretch",
            height=400,
        )
