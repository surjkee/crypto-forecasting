# ui/components/sidebar.py

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from ui.constants import TRACKED_COINS

# ключі та лейбли вкладок
NAV_ITEMS = [
    ("data", "📊 Data"),
    ("features", "🧩 Features"),
    ("forecast", "🔮 Forecast"),
    ("debug", "🛠 Debugging"),
]


def render_sidebar() -> None:
    """
    Малює лівий сайдбар і оновлює st.session_state:
    - active_tab
    - selected_coin_id / selected_coin_label
    - time_range_hours
    - auto_refresh_enabled / автооновлення інтервал
    """

    
    with st.sidebar:

        st.markdown(
    """
<style>

/* --- Overlay sidebar --- */
[data-testid="stSidebar"] {
    position: fixed !important;
    top: 0;
    bottom: 0;
    left: 0;
    width: 300px !important;

    transform: translateX(-125%) !important;
    opacity: 0;
    pointer-events: none;
    transition: transform 1s ease;
}

/* Відкрита панель */
[data-testid="stSidebar"][aria-expanded="true"] {
    transform: translateX(0%) !important;
    transition: transform 0.6s ease;
    pointer-events: auto;
    opacity: 1;
}

/* Напівпрозорий фон основних контейнерів усередині */
[data-testid="stSidebar"],
[data-testid="stSidebar"] [data-testid="stSidebarContent"],
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    background: rgba(14, 12, 23, 0) !important;  /* темне скло */
    backdrop-filter: blur(6px);
}

/* Основний контент не зсуваємо */
[data-testid="stAppViewContainer"] {
    margin-left: 0 !important;
    padding-left: 0 !important;
}

/* Потенційний оверлей робимо прозорим */
[data-testid="stSidebarOverlay"] {
    background: transparent !important;
}

/* Прибрати ВСІ вертикальні скролбари в sidebar’і */

/* Для WebKit (Chrome / Edge / Safari) */
[data-testid="stSidebar"] *::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
}

/* Для Firefox */
[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    scrollbar-width: none !important;
}

</style>
    """,
    unsafe_allow_html=True,
)

        # ---------- CSS ДЛЯ МЕНЮ-НАВІГАЦІЇ (radio без кружків) ----------
        st.markdown(
            """
<style>

/* ============ СПІЛЬНЕ ДЛЯ УСІХ ПУНКТІВ ============ */

/* Ховаємо нативні кружки radio */
[data-testid="stSidebar"] input[type="radio"] {
    display: none !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label > div:nth-child(1) {
    display: none !important;
}

/* Ховаємо квадратик та input чекбокса */
[data-testid="stSidebar"] div[data-testid="stCheckbox"] label > span {
    display: none !important;
}
[data-testid="stSidebar"] div[data-testid="stCheckbox"] input[type="checkbox"] {
    display: none !important;
}

/* Вирівнюємо сам чекбокс-контейнер з радіогрупою */
[data-testid="stSidebar"] div[data-testid="stCheckbox"] {
    padding-left: 0 !important;
    padding-right: 0 !important;
}

/* Базовий стиль рядків меню (і вкладки, і Налаштування) */
[data-testid="stSidebar"] div[role="radiogroup"] > label,
[data-testid="stSidebar"] div[data-testid="stCheckbox"] > label {
    width: 17rem;
    /*display: flex;*/
    align-items: center;
    padding: 0.25rem 0.15rem 0 0;
    cursor: pointer;
    margin: 0.15rem 0;
    opacity: 1;
}

/* Пігулка з текстом (radio: div:nth-child(3), checkbox: div:nth-child(3)) */
[data-testid="stSidebar"] div[role="radiogroup"]
    label[data-baseweb="radio"] > div:nth-child(3),
[data-testid="stSidebar"] div[data-testid="stCheckbox"]
    > label > div:nth-child(3) {

    flex: 1;
    display: flex;
    align-items: center;
    gap: 0.55rem;

    padding: 0.70rem 1.8rem;
    border-radius: 16px;

    opacity: 0.78;
    transition: 0.18s ease;
    /* Якщо дуже хочеш більший текст – раскоментуй:
    font-size: 1.05rem !important;
    */
}

/* ============ АКТИВНІ СТАНИ ============ */

/* Активна вкладка (Data/Features/Forecast/Debugging) */
[data-testid="stSidebar"] div[role="radiogroup"]
    label[data-baseweb="radio"] > input:checked + div {

    background: linear-gradient(90deg, #336dff 0%, #1e2533 55%, #1e2533 100%);
    box-shadow: 0 0 0 1px rgba(255,255,255,0.14);
    opacity: 1.0;
    /*transform: translateX(2px);*/
}

/* Активна кнопка "Налаштування" */
[data-testid="stSidebar"] div[data-testid="stCheckbox"]
    > label > input:checked + div:nth-child(3) {

    background: linear-gradient(90deg, #316dff 0%, #1e2433 55%, #1e2433 100%);
    box-shadow: 0 0 0 1px rgba(255,255,255,0.14);
    opacity: 1.0;
    transform: translateX(2px);
}

/* ============ HOVER ============ */

/* Hover НЕактивних пунктів */
[data-testid="stSidebar"] div[role="radiogroup"]
    label[data-baseweb="radio"]:hover > input:not(:checked) + div,
[data-testid="stSidebar"] div[data-testid="stCheckbox"]
    > label:hover > input:not(:checked) + div:nth-child(3) {

    background: rgba(255,255,255,0.05);
    box-shadow: 0 0 0 1px rgba(255,255,255,0.2);
    opacity: 0.8;
}

/* Hover активних */
[data-testid="stSidebar"] div[role="radiogroup"]
    label[data-baseweb="radio"]:hover > input:checked + div,
[data-testid="stSidebar"] div[data-testid="stCheckbox"]
    > label:hover > input:checked + div:nth-child(3) {

    box-shadow: 0 0 0 1px rgba(255,255,255,0.2);
    opacity: 0.8;
}

</style>

            """,
            unsafe_allow_html=True,
        )

        
        # ---------- ВИБІР МОНЕТИ ----------
        coin_labels = [label for label, _ in TRACKED_COINS]
        coin_ids = [cid for _, cid in TRACKED_COINS]

        # дефолт – bitcoin, якщо є
        default_coin_id = "bitcoin" if "bitcoin" in coin_ids else coin_ids[0]
        default_coin_label = coin_labels[coin_ids.index(default_coin_id)]

        current_coin_id = st.session_state.get("selected_coin_id", default_coin_id)
        if current_coin_id not in coin_ids:
            current_coin_id = default_coin_id

        current_index = coin_ids.index(current_coin_id)

        selected_coin_label = st.selectbox(
            "Choose Coin:",
            options=coin_labels,
            index=current_index,
            key="sidebar_coin_select",
        )
        selected_coin_id = coin_ids[coin_labels.index(selected_coin_label)]

        st.session_state["selected_coin_id"] = selected_coin_id
        st.session_state["selected_coin_label"] = selected_coin_label

        st.markdown("---")
        
        # ---------- НАВІГАЦІЯ ПО РОЗДІЛАХ ----------
        if "active_tab" not in st.session_state:
            st.session_state["active_tab"] = "data"

        nav_keys = [k for k, _ in NAV_ITEMS]
        nav_labels = [lbl for _, lbl in NAV_ITEMS]

        current_tab_key = st.session_state.get("active_tab", "data")
        if current_tab_key not in nav_keys:
            current_tab_key = "data"

        default_index = nav_keys.index(current_tab_key)

        selected_label = st.radio(
            "Розділ",
            options=nav_labels,
            index=default_index,
            key="nav_tab",
            label_visibility="collapsed",
        )

        # оновлюємо active_tab
        selected_key = nav_keys[nav_labels.index(selected_label)]
        st.session_state["active_tab"] = selected_key

        # Кнопка-перемикач налаштувань
        settings_open = st.checkbox(
            "⚙️ Settings",
            key="sidebar_settings_open",
            value=True,
        )

        st.markdown("---")

        if settings_open:
            # ---------- ЧАСОВІ НАЛАШТУВАННЯ ----------
            st.subheader("⏱ Часові налаштування")

            time_range_label = st.selectbox(
                "Період даних",
                options=[
                    "24 години",
                    "3 дні",
                    "7 днів",
                    "30 днів",
                    "Увесь період",
                ],
                index=2,  # дефолт – 7 днів
                key="time_range_label",
            )

            time_range_hours_map = {
                "24 години": 24,
                "3 дні": 24 * 3,
                "7 днів": 24 * 7,
                "30 днів": 24 * 30,
                "Увесь період": None,
            }
            st.session_state["time_range_hours"] = time_range_hours_map[time_range_label]

            st.markdown("---")

            
            # ---------- АВТООНОВЛЕННЯ ----------
            st.subheader("🔄 Автооновлення")

            auto_refresh_enabled = st.checkbox(
                "Увімкнути автооновлення",
                value=False,
                key="auto_refresh_enabled",
            )

            refresh_interval_label = st.selectbox(
                "Інтервал автооновлення",
                options=["30 секунд", "1 хвилина", "5 хвилин"],
                index=1,
                key="auto_refresh_interval_label",
            )

            if auto_refresh_enabled:
                interval_ms_map = {
                    "30 секунд": 30_000,
                    "1 хвилина": 60_000,
                    "5 хвилин": 5 * 60_000,
                }
                interval_ms = interval_ms_map[refresh_interval_label]

                st_autorefresh(interval=interval_ms, key="global_autorefresh")