"""Shared UI tokens and responsive styles for both session themes."""
import matplotlib.pyplot as plt
import streamlit as st

PALETTES = {
    "Dark Mode": {"background": "#101319", "card-background": "#191E27",
                  "text-primary": "#F3F5F8", "text-secondary": "#A5AFBF",
                  "border": "#303846", "accent": "#E45454", "input": "#202733",
                  "shadow": "0 12px 30px rgba(0,0,0,.18)", "theme-toggle-icon": "#E8CE8B", "gain": "#72C7A6", "loss": "#F18C91"},
    "Light Mode": {"background": "#F3F5F8", "card-background": "#FFFFFF",
                   "text-primary": "#172131", "text-secondary": "#637083",
                   "border": "#DCE2EB", "accent": "#BD353A", "input": "#FFFFFF",
                   "shadow": "0 8px 24px rgba(23,33,49,.06)", "theme-toggle-icon": "#172131", "gain": "#207553", "loss": "#B53642"},
}

CSS = """
:root { color-scheme: SCHEME; TOKENS }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {background:var(--background);color:var(--text-primary)}
[data-testid="stMainBlockContainer"] {max-width:1440px;padding:5rem 2.6rem 4rem}
[data-testid="stSidebar"] {background:var(--card-background);border-right:1px solid var(--border)}
[data-testid="stSidebar"] > div {padding-top:1.6rem}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label,.stApp li {color:var(--text-primary)}
.stApp h1 {font-size:2rem;font-weight:750;letter-spacing:-.055em;line-height:1.15}
.stApp h2 {font-size:1.45rem;letter-spacing:-.035em}
.stApp h3 {font-size:1.1rem;font-weight:650}
.stApp [data-testid="stCaptionContainer"] p {color:var(--text-secondary)}
.stApp [data-testid="stWidgetLabel"] p {font-size:.82rem;font-weight:600;color:var(--text-secondary)}
.stApp a {color:var(--accent)}
.stApp [data-baseweb="select"] > div,.stApp input,.stApp [data-baseweb="input"],.stApp [data-baseweb="base-input"] {background:var(--input);color:var(--text-primary);border-color:var(--border);border-radius:10px}
.stApp [data-testid="stSelectbox"] [role="group"] {background:var(--input);border-color:var(--border);border-radius:10px}
.stApp [data-testid="stSelectbox"] input {background:transparent;border:0;border-radius:0;color:var(--text-primary)}
.stApp [data-testid="stSelectbox"] button {background:transparent;color:var(--text-primary);border:0}
[data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"] {background:var(--card-background)!important;color:var(--text-primary)!important}
[role="option"] {color:var(--text-primary)!important;background:var(--card-background)}
[role="option"]:hover {background:var(--input)!important}
.stApp [data-testid="stButton"] button,.stApp [data-testid="stDownloadButton"] button {border:1px solid var(--border);border-radius:10px;background:var(--card-background);color:var(--text-primary);min-height:42px;font-weight:600}
.stApp [data-testid="stButton"] button[kind="primary"] {background:var(--accent);color:#fff;border-color:var(--accent)}
.stApp [data-testid="stButton"] button[kind="primary"] p {color:#fff}
.stApp [data-testid="stButton"] button:hover {border-color:var(--accent)}
.stApp [data-testid="stMetric"] {background:var(--card-background);padding:1.25rem;border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow)}
.stApp [data-testid="stMetricValue"] {color:var(--text-primary);font-size:1.6rem}
.stApp [data-testid="stExpander"] {background:var(--card-background);border:1px solid var(--border);border-radius:12px}
.stApp [data-testid="stExpander"] summary {color:var(--text-primary)}
.stApp [data-testid="stDataFrame"],.stApp [data-testid="stDataEditor"] {border:1px solid var(--border);border-radius:12px;overflow:hidden}
.stApp [data-testid="stTabs"] [role="tablist"] {gap:1.5rem;border-bottom:1px solid var(--border)}
.stApp [role="tab"] {color:var(--text-secondary)}
.stApp [role="tab"][aria-selected="true"] {color:var(--accent)}
.stApp [data-testid="stRadio"] div[role="radiogroup"] {gap:1rem}
.stApp [data-testid="stMain"] [data-testid="stRadio"] label {padding:10px 16px;border:1px solid var(--border);border-radius:10px;background:var(--card-background)}
.stApp [data-testid="stMain"] [data-testid="stRadio"] label:has(input:checked) {border-color:var(--accent)}
.stApp [data-testid="stMain"] [data-testid="stRadio"] label > div > div:first-child {display:none}
.stApp [data-testid="stMain"] [data-testid="stRadio"] label[data-selected="true"] p {color:var(--accent)}
.stApp [data-testid="stAlert"] {border-radius:12px}
.brand {display:flex;align-items:center;gap:14px;margin-bottom:8px}
.brand-mark {width:42px;height:42px;border-radius:11px;background:var(--accent);color:#fff;display:grid;place-items:center;font-size:17px;font-weight:850;letter-spacing:-2px;transform:skew(-8deg)}
.brand-title {font-size:1.6rem;font-weight:750;letter-spacing:-.05em;color:var(--text-primary)}
.brand-subtitle {font-size:.88rem;color:var(--text-secondary);margin-top:5px}
.eyebrow {font-size:.7rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin-bottom:8px}
.section-heading {margin:1.5rem 0 1rem}
.section-heading h2 {margin:0;color:var(--text-primary)}
.section-heading p {font-size:.9rem;color:var(--text-secondary);margin:.4rem 0 0}
.event-card {background:var(--card-background);border:1px solid var(--border);border-radius:16px;padding:1.6rem 1.8rem;margin:1rem 0;box-shadow:var(--shadow);border-left:4px solid var(--accent)}
.event-card h2 {margin:0 0 .55rem;font-size:1.75rem;color:var(--text-primary)}
.event-details {display:flex;flex-wrap:wrap;gap:12px 24px;color:var(--text-secondary);font-size:.82rem}
.driver-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px;margin:12px 0 20px}
.driver-grid.podium {grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}
.driver-card {position:relative;background:var(--card-background);color:var(--text-primary);border:1px solid var(--border);border-top:3px solid var(--team-accent);border-radius:16px;overflow:hidden;box-shadow:var(--shadow);transition:transform .16s ease,box-shadow .16s ease;outline-offset:4px}
.driver-card:hover,.driver-card:focus-within {transform:translateY(-3px);box-shadow:0 14px 34px rgba(0,0,0,.15)}
.driver-card:focus {outline:2px solid var(--accent)}
.card-top {display:flex;justify-content:space-between;align-items:center;padding:14px 18px 0;position:relative;z-index:2}
.position {font-weight:800;font-size:1.25rem;letter-spacing:-.05em}
.podium .position {font-size:2rem}
.driver-code {font-size:.72rem;font-weight:700;letter-spacing:.12em;color:var(--text-secondary)}
.portrait {height:135px;position:relative;overflow:hidden;margin-top:5px;background:var(--card-background)}
.podium .portrait {height:195px}
.portrait img {width:100%;height:440px;object-fit:contain;object-position:center top;position:absolute;inset:0;color:var(--text-secondary);font-size:.7rem}
.podium .portrait img {height:600px}
.comparison .portrait img {height:520px}
.avatar-fallback {position:absolute;inset:0;display:grid;place-items:center;color:var(--text-secondary);opacity:.07}
.portrait:has(img) .avatar-fallback {opacity:.025}
.avatar-fallback svg {height:85%;width:auto}
.card-copy {padding:16px 18px 18px;border-top:1px solid var(--border)}
.card-name {font-weight:700;font-size:1rem;letter-spacing:-.02em;margin-bottom:4px}
.card-team {color:var(--text-secondary);font-size:.8rem}
.card-score {display:flex;justify-content:space-between;margin-top:14px;font-size:.78rem;color:var(--text-secondary)}
.card-score strong {color:var(--text-primary);font-variant-numeric:tabular-nums;font-weight:650}
.driver-stats {position:absolute;inset:0;z-index:3;background:var(--card-background);padding:24px 20px;opacity:0;pointer-events:none;transition:opacity .15s ease;display:flex;flex-direction:column;justify-content:center}
.driver-card:hover .driver-stats,.driver-card:focus .driver-stats {opacity:1}
.stats-name {font-weight:750;font-size:1.15rem;margin-bottom:16px}
.stats-row {display:flex;justify-content:space-between;font-size:.86rem;padding:8px 0;border-bottom:1px solid var(--border);color:var(--text-secondary)}
.stats-row strong {color:var(--text-primary)}
.stats-source {font-size:.66rem;color:var(--text-secondary);margin-top:12px;line-height:1.5}
.comparison {display:grid;grid-template-columns:minmax(0,1fr) 70px minmax(0,1fr);max-width:850px;margin:1rem auto 2rem;align-items:center;gap:20px}
.comparison .portrait {height:180px}
.versus {text-align:center;font-weight:750;font-size:.85rem;letter-spacing:.12em;color:var(--text-secondary)}
.exact-table {width:100%;border-collapse:collapse;color:var(--text-primary);font-size:.87rem}
.exact-table th {text-align:left;color:var(--text-secondary);font-weight:600;border-bottom:1px solid var(--border);padding:12px}
.exact-table td {padding:12px;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums}
.table-wrap {overflow-x:auto;background:var(--card-background);border-radius:12px}
.site-foot {margin-top:3rem;padding-top:1.1rem;border-top:1px solid var(--border);font-size:.72rem;color:var(--text-secondary)}
.race-card-details {margin-top:14px;border-top:1px solid var(--border);padding-top:8px}
.race-detail {display:flex;justify-content:space-between;gap:8px;font-size:.75rem;padding:4px 0;color:var(--text-secondary)}
.race-detail strong {color:var(--text-primary);text-align:right;font-weight:600;overflow-wrap:anywhere}
.race-detail .gain {color:var(--gain)} .race-detail .loss {color:var(--loss)}
.race-podium .driver-card:first-child {border-top-width:5px;box-shadow:0 8px 28px rgba(0,0,0,.12)}
.race-classification {grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}
/* Only this keyed container floats; ordinary application buttons keep their styles. */
.stApp .st-key-floating_theme_control {position:fixed;right:24px;bottom:24px;width:52px;height:52px;z-index:1000}
.stApp .st-key-floating_theme_control [data-testid="stButton"] button {
    width:52px;height:52px;min-height:52px;min-width:52px;padding:0;gap:0;border-radius:50%;
    border:1px solid var(--border);background:var(--card-background);color:var(--theme-toggle-icon);
    box-shadow:0 4px 16px rgba(0,0,0,.16);transition:all .2s ease;display:flex;align-items:center;justify-content:center;
}
.stApp .st-key-floating_theme_control button [data-testid="stMarkdownContainer"] {
    position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap;
}
.stApp .st-key-floating_theme_control button:hover {transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,0,0,.24);border-color:var(--text-secondary)}
.stApp .st-key-floating_theme_control button:focus-visible {outline:2px solid var(--accent);outline-offset:4px}
@media(max-width:540px){.stApp .st-key-floating_theme_control{right:16px;bottom:max(16px,env(safe-area-inset-bottom))}}
@media(prefers-reduced-motion:reduce){.stApp .st-key-floating_theme_control button{transition:none}.stApp .st-key-floating_theme_control button:hover{transform:none}}
@media(max-width:800px){[data-testid="stMainBlockContainer"]{padding:4.5rem 1rem 3rem}.brand-title{font-size:1.3rem}.brand-subtitle{font-size:.78rem}.podium .portrait{height:140px}.comparison{gap:10px;grid-template-columns:minmax(0,1fr) 28px minmax(0,1fr)}.comparison .portrait{height:140px}.event-card{padding:1.2rem}}
@media(max-width:540px){.driver-grid.podium{grid-template-columns:1fr}.podium .portrait{height:190px}.driver-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.race-classification{grid-template-columns:1fr}.card-copy{padding:12px}.card-name{font-size:.9rem}.comparison .portrait{height:100px}.driver-stats{padding:14px}.stats-row{font-size:.75rem}.brand-mark{width:34px;height:34px}}
@media(prefers-reduced-motion:reduce){.driver-card,.driver-stats{transition:none}.driver-card:hover,.driver-card:focus-within{transform:none}}
@media(hover:none){.driver-card:focus .driver-stats{opacity:1}.driver-card:hover:not(:focus) .driver-stats{opacity:0}}
"""


def apply_theme():
    theme = st.session_state.get("ui_theme", "Dark Mode")
    palette = PALETTES[theme]
    tokens = ";".join(f"--{key}:{value}" for key, value in palette.items()) + ";"
    css = CSS.replace("TOKENS", tokens).replace("SCHEME", "dark" if theme == "Dark Mode" else "light")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    plt.style.use("dark_background" if theme == "Dark Mode" else "default")
    plt.rcParams.update({"figure.facecolor": palette["card-background"], "axes.facecolor": palette["card-background"],
                         "text.color": palette["text-primary"], "axes.labelcolor": palette["text-secondary"],
                         "xtick.color": palette["text-secondary"], "ytick.color": palette["text-secondary"],
                         "axes.edgecolor": palette["border"], "grid.color": palette["border"]})


def styled_dataframe(frame):
    palette = PALETTES[st.session_state.get("ui_theme", "Dark Mode")]
    return frame.style.set_properties(**{"background-color": palette["card-background"], "color": palette["text-primary"]})
