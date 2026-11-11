
from pathlib import Path
import json
import sqlite3
import hashlib
import secrets
import hmac

import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"

MODEL_PATHS = {
    "PaySim": ROOT / "models/paysim/evaluation.json",
    "IEEE-CIS": ROOT / "models/ieee_cis/evaluation.json",
}

MODEL_ARTIFACTS = {
    "PaySim": ROOT / "models/paysim/logistic_regression.joblib",
    "IEEE-CIS": ROOT / "models/ieee_cis/logistic_regression.joblib",
}

DATASETS = {
    "MoMTSim V1": {
        "transactions": 1_720_181,
        "fraud": 175_518,
        "fraud_rate": 10.2035,
        "domain": "Mobile money",
        "description": (
            "A synthetic mobile-money dataset containing transaction "
            "types, amounts, account balances and fraud labels."
        ),
        "status": "Dataset analysis and evaluation completed",
        "limitation": (
            "Fraud labels are concentrated in TRANSFER transactions. "
            "Synthetic generation patterns may influence model results."
        ),
    },
    "MoMTSim V2": {
        "transactions": 4_225_958,
        "fraud": 2_233_118,
        "fraud_rate": 52.8429,
        "domain": "Mobile money",
        "description": (
            "A second synthetic mobile-money dataset with a different "
            "transaction population and fraud prevalence."
        ),
        "status": "Dataset analysis and provenance investigation completed",
        "limitation": (
            "The high fraud prevalence and synthetic patterns make "
            "generalization to real-world fraud uncertain."
        ),
    },
    "PaySim": {
        "transactions": 6_362_620,
        "fraud": 8_213,
        "fraud_rate": 0.1291,
        "domain": "Mobile money",
        "description": (
            "A simulated mobile-money dataset containing CASH_IN, "
            "CASH_OUT, PAYMENT, TRANSFER and DEBIT transactions."
        ),
        "status": "Chronological baseline model trained",
        "limitation": (
            "The dataset has severe class imbalance and changes in "
            "fraud prevalence over time."
        ),
    },
    "IEEE-CIS": {
        "transactions": 590_540,
        "fraud": 20_663,
        "fraud_rate": 3.4995,
        "domain": "Card payments",
        "description": (
            "A card transaction fraud dataset containing transaction "
            "features and a separate identity information table."
        ),
        "status": "Chronological baseline model trained",
        "limitation": (
            "Many identity fields contain missing values. Test performance "
            "declines compared with validation performance."
        ),
    },
}

PAYSIM_TEMPORAL = {
    "Period": [
        "Steps 1–100",
        "Steps 101–200",
        "Steps 201–300",
        "Steps 301–400",
        "Steps 401–500",
        "Steps 501–600",
        "Steps 601–700",
        "Steps 701–743",
    ],
    "Fraud rate (%)": [
        0.112,
        0.080,
        0.068,
        0.063,
        0.395,
        0.533,
        1.205,
        4.290,
    ],
}


# ============================================================
# 2. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Intelligence | Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 3. NIGHT / DAY THEME
# ============================================================

if "night_mode" not in st.session_state:
    st.session_state.night_mode = True

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Overview"

_, theme_col = st.columns([5, 1])

with theme_col:
    st.toggle(
        "Night Mode",
        key="night_mode",
        help="Switch between dark and light dashboard themes.",
    )

if st.session_state.night_mode:
    colors = {
        "bg": "#0B1120",
        "card": "#172337",
        "surface": "#202E43",
        "text": "#F8FAFC",
        "muted": "#CBD5E1",
        "border": "#35465E",
        "accent": "#38BDF8",
        "hero": "#172B46",
    }
else:
    colors = {
        "bg": "#F3F6FB",
        "card": "#FFFFFF",
        "surface": "#E8EEF6",
        "text": "#172033",
        "muted": "#475569",
        "border": "#CBD5E1",
        "accent": "#0369A1",
        "hero": "#DBEAFE",
    }

st.markdown(
    f"""
    <style>
    :root {{
        --app-bg: {colors["bg"]};
        --app-card: {colors["card"]};
        --app-surface: {colors["surface"]};
        --app-text: {colors["text"]};
        --app-muted: {colors["muted"]};
        --app-border: {colors["border"]};
        --app-accent: {colors["accent"]};
        --app-hero: {colors["hero"]};
    }}

    /* APPLICATION BACKGROUND */

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {{
        background-color: var(--app-bg) !important;
        color: var(--app-text) !important;
    }}

    [data-testid="stHeader"] {{
        background-color: var(--app-bg) !important;
    }}

    /* GENERAL TYPOGRAPHY */

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp p,
    .stApp li,
    .stApp label,
    .stApp [data-testid="stMarkdownContainer"] {{
        color: var(--app-text) !important;
    }}

    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p {{
        color: var(--app-muted) !important;
    }}

    .stApp h1 {{
        font-size: 34px !important;
        font-weight: 800 !important;
        line-height: 1.25 !important;
    }}

    .stApp h2 {{
        font-size: 27px !important;
        font-weight: 750 !important;
    }}

    .stApp h3 {{
        font-size: 22px !important;
        font-weight: 700 !important;
    }}

    /* MAIN DASHBOARD TITLE */

    .dashboard-title {{
        color: var(--app-text) !important;
        font-size: 38px !important;
        font-weight: 850 !important;
        letter-spacing: 0.5px !important;
        line-height: 1.3 !important;
        margin: 4px 0 16px 0 !important;
    }}

    /* HERO BANNER */

    .hero-box {{
        background: var(--app-hero) !important;
        border: 1px solid var(--app-border) !important;
        border-radius: 18px !important;
        padding: 34px !important;
        margin: 12px 0 22px 0 !important;
        animation: fadeUp 0.5s ease-out;
    }}

    .hero-box h1 {{
        color: var(--app-text) !important;
        font-size: 46px !important;
        font-weight: 850 !important;
        line-height: 1.2 !important;
        margin: 12px 0 !important;
    }}

    .hero-label {{
        color: var(--app-muted) !important;
        font-size: 13px !important;
        font-weight: 750 !important;
        letter-spacing: 2px !important;
    }}

    .hero-description {{
        color: var(--app-text) !important;
        font-size: 18px !important;
        line-height: 1.8 !important;
        max-width: 900px;
    }}

    /* METRIC CARDS */

    [data-testid="stMetric"] {{
        background: var(--app-card) !important;
        border: 1px solid var(--app-border) !important;
        border-radius: 14px !important;
        padding: 20px !important;
        animation: fadeUp 0.5s ease-out;
    }}

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {{
        color: var(--app-text) !important;
    }}

    /* NAVIGATION */

    [data-testid="stRadio"] [role="radiogroup"] {{
        gap: 8px !important;
        flex-wrap: wrap !important;
    }}

    [data-testid="stRadio"] [role="radiogroup"] label {{
        background: var(--app-surface) !important;
        border: 1px solid var(--app-border) !important;
        border-radius: 9px !important;
        padding: 8px 13px !important;
    }}

    [data-testid="stRadio"] [role="radiogroup"] label p,
    [data-testid="stRadio"] [role="radiogroup"] label span {{
        color: var(--app-text) !important;
    }}

    /* BUTTONS */

    .stButton > button {{
        background: var(--app-surface) !important;
        color: var(--app-text) !important;
        border: 1px solid var(--app-border) !important;
        border-radius: 10px !important;
    }}

    .stButton > button:hover {{
        border-color: var(--app-accent) !important;
        color: var(--app-accent) !important;
    }}

    /* INPUTS AND SELECT BOXES */

    .stApp input,
    .stApp textarea,
    .stApp div[data-baseweb="select"] > div {{
        background: var(--app-card) !important;
        color: var(--app-text) !important;
        border-color: var(--app-border) !important;
    }}

    .stApp input::placeholder,
    .stApp textarea::placeholder {{
        color: var(--app-muted) !important;
    }}

    [data-baseweb="popover"],
    [data-baseweb="popover"] *,
    [role="option"] {{
        color: var(--app-text) !important;
    }}

    /* CONTAINERS AND EXPANDERS */

    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stExpander"] {{
        background: var(--app-card) !important;
        border: 1px solid var(--app-border) !important;
        border-radius: 12px !important;
    }}

    /* SIDEBAR */

    [data-testid="stSidebar"] {{
        background: var(--app-card) !important;
    }}

    /* TABLES */

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {{
        border: 1px solid var(--app-border) !important;
    }}

    /* ALERTS AND DIVIDERS */

    [data-testid="stAlert"] {{
        background: var(--app-card) !important;
        color: var(--app-text) !important;
    }}

    hr {{
        border-color: var(--app-border) !important;
    }}

    /* FOOTER */

    .app-footer {{
        color: var(--app-muted) !important;
        text-align: center;
        padding: 18px;
        font-size: 17px;
        font-weight: 800;
        letter-spacing: 2px;
    }}

    /* ANIMATIONS AND RESPONSIVE DESIGN */

    @keyframes fadeUp {{
        from {{
            opacity: 0;
            transform: translateY(8px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @media (max-width: 768px) {{
        .dashboard-title {{
            font-size: 29px !important;
        }}

        .hero-box h1 {{
            font-size: 34px !important;
        }}
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. AUTHENTICATION, REGISTRATION AND ADMIN MANAGEMENT
# ============================================================

AUTH_DB = ROOT / "auth_users.db"


def get_connection():
    return sqlite3.connect(AUTH_DB)


def initialize_database():

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()


# ------------------------------------------------------------
# PASSWORD HASHING
# ------------------------------------------------------------

def hash_password(password, salt=None):

    if salt is None:
        salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        600000,
    ).hex()

    return password_hash, salt


def verify_password(password, stored_hash, salt):

    calculated_hash, _ = hash_password(password, salt)

    return hmac.compare_digest(calculated_hash, stored_hash)


# ------------------------------------------------------------
# ADMIN ACCOUNT INITIALIZATION
# ------------------------------------------------------------

def initialize_admin():

    try:
        admin_username = st.secrets["auth"]["username"]
        admin_password = st.secrets["auth"]["password"]

    except (KeyError, FileNotFoundError):
        st.error(
            "Admin credentials are missing. "
            "Configure .streamlit/secrets.toml."
        )
        st.stop()

    with get_connection() as conn:

        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (admin_username,),
        ).fetchone()

        if existing is None:

            password_hash, salt = hash_password(admin_password)

            conn.execute("""
                INSERT INTO users
                (username, email, password_hash, salt, role, status)
                VALUES (?, ?, ?, ?, 'admin', 'approved')
            """, (
                admin_username,
                "admin@local",
                password_hash,
                salt,
            ))

            conn.commit()


# ------------------------------------------------------------
# USER REGISTRATION
# ------------------------------------------------------------

def register_user(username, email, password, confirm_password):

    username = username.strip()
    email = email.strip().lower()

    if not username or not email or not password:
        return False, "Please complete all fields."

    if len(username) < 3:
        return False, "Username must contain at least 3 characters."

    if len(password) < 10:
        return False, "Password must contain at least 10 characters."

    if password != confirm_password:
        return False, "Passwords do not match."

    if "@" not in email or "." not in email:
        return False, "Please enter a valid email address."

    password_hash, salt = hash_password(password)

    try:

        with get_connection() as conn:

            conn.execute("""
                INSERT INTO users
                (username, email, password_hash, salt, role, status)
                VALUES (?, ?, ?, ?, 'user', 'pending')
            """, (
                username,
                email,
                password_hash,
                salt,
            ))

            conn.commit()

        return True, "Account created. Please wait for admin approval."

    except sqlite3.IntegrityError:
        return False, "Username or email already exists."


# ------------------------------------------------------------
# USER LOGIN
# ------------------------------------------------------------

def authenticate_user(username, password):

    with get_connection() as conn:

        user = conn.execute("""
            SELECT id, username, email, password_hash, salt, role, status
            FROM users
            WHERE username = ? COLLATE NOCASE
        """, (username.strip(),)).fetchone()

    if user is None:
        return False, "Invalid username or password."

    user_id, name, email, stored_hash, salt, role, status = user

    if not verify_password(password, stored_hash, salt):
        return False, "Invalid username or password."

    if status == "pending":
        return False, "Your account is awaiting admin approval."

    if status != "approved":
        return False, "Your account is inactive. Contact the administrator."

    st.session_state["authenticated"] = True
    st.session_state["user_id"] = user_id
    st.session_state["username"] = name
    st.session_state["user_role"] = role

    return True, "Login successful."


# ------------------------------------------------------------
# LOGIN AND CREATE ACCOUNT PAGE
# ------------------------------------------------------------

def show_auth_page():

    st.markdown(
        """
        <div style="text-align:center;padding:30px 10px;">
            <h1 style="font-size:38px;font-weight:800;">
                🛡️ FRAUD INTELLIGENCE
            </h1>
            <p style="font-size:17px;">
                Multi-Channel Financial Fraud Detection System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 1.2, 1])

    with center:

        login_tab, register_tab = st.tabs(
            ["Sign In", "Create Account"]
        )

        # SIGN IN
        with login_tab:

            st.subheader("Welcome Back")
            st.caption("Sign in to access your dashboard.")

            with st.form("login_form"):

                username = st.text_input(
                    "Username",
                    placeholder="Enter your username",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                )

                submitted = st.form_submit_button(
                    "Sign In",
                    use_container_width=True,
                )

                if submitted:

                    success, message = authenticate_user(
                        username,
                        password,
                    )

                    if success:
                        st.rerun()
                    else:
                        st.error(message)

        # CREATE ACCOUNT
        with register_tab:

            st.subheader("Create Your Account")
            st.caption(
                "Register to request access to the research dashboard."
            )

            with st.form("register_form"):

                new_username = st.text_input("Choose Username")
                new_email = st.text_input("Email Address")

                new_password = st.text_input(
                    "Create Password",
                    type="password",
                )

                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                )

                register_button = st.form_submit_button(
                    "Create Account",
                    use_container_width=True,
                )

                if register_button:

                    success, message = register_user(
                        new_username,
                        new_email,
                        new_password,
                        confirm_password,
                    )

                    if success:
                        st.success(message)
                    else:
                        st.error(message)

    st.write("")
    st.caption(
        "Authorized access only · Financial Analytics Research"
    )


# ------------------------------------------------------------
# ADMIN USER MANAGEMENT
# ------------------------------------------------------------

def show_user_management():

    st.title("User Management")
    st.caption("Administrator control panel")

    if st.session_state.get("user_role") != "admin":
        st.error("Access denied. Administrator privileges required.")
        st.stop()

    with get_connection() as conn:

        users = conn.execute("""
            SELECT id, username, email, role, status, created_at
            FROM users
            ORDER BY created_at DESC
        """).fetchall()

    if not users:
        st.info("No registered users.")
        return

    pending = [u for u in users if u[4] == "pending"]
    approved = [u for u in users if u[4] == "approved"]
    inactive = [u for u in users if u[4] == "inactive"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Users", len(users))
    col2.metric("Pending", len(pending))
    col3.metric("Approved", len(approved))
    col4.metric("Inactive", len(inactive))

    st.divider()

    st.subheader("Registered Users")

    for user in users:

        user_id, username, email, role, status, created_at = user

        with st.expander(
            f"{username} · {status.upper()} · {role.upper()}"
        ):

            st.write("**Username:**", username)
            st.write("**Email:**", email)
            st.write("**Role:**", role)
            st.write("**Status:**", status)
            st.write("**Registered:**", created_at)

            if role == "admin":
                st.info("Administrator account")
                continue

            col_a, col_b = st.columns(2)

            if status == "pending":

                with col_a:

                    if st.button(
                        "Approve",
                        key=f"approve_{user_id}",
                        use_container_width=True,
                    ):

                        with get_connection() as conn:
                            conn.execute(
                                "UPDATE users SET status = 'approved' "
                                "WHERE id = ? AND role = 'user'",
                                (user_id,),
                            )
                            conn.commit()

                        st.success(f"{username} approved.")
                        st.rerun()

            if status == "approved":

                with col_a:

                    if st.button(
                        "Deactivate",
                        key=f"deactivate_{user_id}",
                        use_container_width=True,
                    ):

                        with get_connection() as conn:
                            conn.execute(
                                "UPDATE users SET status = 'inactive' "
                                "WHERE id = ? AND role = 'user'",
                                (user_id,),
                            )
                            conn.commit()

                        st.success(f"{username} deactivated.")
                        st.rerun()

            if status == "inactive":

                with col_a:

                    if st.button(
                        "Reactivate",
                        key=f"reactivate_{user_id}",
                        use_container_width=True,
                    ):

                        with get_connection() as conn:
                            conn.execute(
                                "UPDATE users SET status = 'approved' "
                                "WHERE id = ? AND role = 'user'",
                                (user_id,),
                            )
                            conn.commit()

                        st.success(f"{username} reactivated.")
                        st.rerun()


# ------------------------------------------------------------
# INITIALIZE AUTHENTICATION
# ------------------------------------------------------------

initialize_database()
initialize_admin()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:

    show_auth_page()
    st.stop()


# ------------------------------------------------------------
# LOGGED-IN USER HEADER AND LOGOUT
# ------------------------------------------------------------

user_col, role_col, logout_col = st.columns([4, 2, 1])

with user_col:
    st.caption(
        f"Signed in as: {st.session_state['username']}"
    )

with role_col:
    st.caption(
        f"Role: {st.session_state['user_role'].upper()}"
    )

with logout_col:

    if st.button("Logout", use_container_width=True):

        for key in [
            "authenticated",
            "user_id",
            "username",
            "user_role",
        ]:
            st.session_state.pop(key, None)

        st.session_state["selected_page"] = "Overview"

        st.rerun()


# ============================================================
# 5. DATA LOADING AND HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_evaluation(path_string):

    path = Path(path_string)

    if not path.exists():
        return None

    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return None


def format_number(value):
    return f"{value:,}"


def get_model_results():

    records = []

    for model_name, path in MODEL_PATHS.items():

        result = load_evaluation(str(path))

        if result is None:
            continue

        test = result.get("test", {})

        records.append({
            "Dataset": model_name,
            "PR-AUC": test.get("average_precision"),
            "ROC-AUC": test.get("roc_auc"),
            "Precision": test.get("precision"),
            "Recall": test.get("recall"),
            "F1": test.get("f1"),
        })

    return pd.DataFrame(records)


def metric_definition(name):

    definitions = {
        "PR-AUC": (
            "Area under the precision-recall curve. Particularly useful "
            "for evaluating fraud detection when fraud cases are rare."
        ),
        "ROC-AUC": (
            "Measures how well a model ranks fraudulent transactions "
            "above legitimate transactions across different thresholds."
        ),
        "Precision": (
            "Of the transactions flagged as fraud, how many were actually "
            "fraudulent?"
        ),
        "Recall": (
            "Of all actual fraud cases, how many did the model detect?"
        ),
        "F1": (
            "A combined measure of precision and recall."
        ),
    }

    return definitions.get(name, "")


# ============================================================
# 6. TOP HEADER
# ============================================================

header_left, header_right = st.columns([2, 1])

with header_left:

    st.markdown(
        '<div class="dashboard-title">🛡️ FRAUD INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

with header_right:

    st.markdown(
        f"""
        <p style="
            text-align:right;
            color:{colors['muted']};
            font-size:13px;
            font-weight:700;
            letter-spacing:2px;
            margin-top:15px;
        ">
            FINANCIAL ANALYTICS RESEARCH
        </p>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 7. TOP NAVIGATION
# ============================================================

PAGES = [
    "Overview",
    "Datasets",
    "Data Quality",
    "Model Lab",
    "Comparison",
    "Reports",
    "Glossary",
    "About",
]

if st.session_state.get("user_role") == "admin":
    PAGES.append("User Management")

# Prevent an invalid saved page selection
# if the current user does not have admin access.

if st.session_state.get("selected_page") not in PAGES:
    st.session_state["selected_page"] = "Overview"

page = st.radio(
    "Navigation",
    PAGES,
    horizontal=True,
    label_visibility="collapsed",
    key="selected_page",
)

st.divider()


# ============================================================
# 8. OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-label">
                FINANCIAL SECURITY · MACHINE LEARNING · ANALYTICS
            </div>
            <h1>Financial Fraud Detection</h1>
            <div class="hero-description">
                Explore financial transaction datasets, understand
                suspicious patterns, and evaluate machine-learning
                models designed to identify potentially fraudulent activity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # PROJECT AT A GLANCE
    # --------------------------------------------------------

    st.subheader("Project at a glance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Datasets investigated", "4")
    col2.metric("PaySim records", "6,362,620")
    col3.metric("IEEE-CIS records", "590,540")
    col4.metric("Saved baseline models", "2")

    st.caption(
        "These are separate datasets. Their transaction counts are "
        "not added together because they represent different populations."
    )

    st.divider()

    # --------------------------------------------------------
    # PROJECT DESCRIPTION
    # --------------------------------------------------------

    st.subheader("What does this project do?")

    st.write(
        "Financial fraud detection is the process of identifying "
        "transactions that may involve unauthorized or deceptive activity."
    )

    st.write(
        "This project investigates historical transaction data and "
        "evaluates machine-learning models that attempt to distinguish "
        "fraudulent transactions from legitimate ones."
    )

    st.write(
        "The dashboard presents dataset characteristics, data-quality "
        "findings, model evaluation results and research documentation."
    )

    st.divider()

    # --------------------------------------------------------
    # RESEARCH WORKFLOW
    # --------------------------------------------------------

    st.subheader("Research workflow")

    workflow = st.columns(4)

    workflow_items = [
        (
            "01 — Data inspection",
            "Understand dataset structure, labels and missing values.",
        ),
        (
            "02 — Data preparation",
            "Prepare features and chronological data partitions.",
        ),
        (
            "03 — Model training",
            "Train classification models using historical examples.",
        ),
        (
            "04 — Evaluation",
            "Measure performance on unseen later records.",
        ),
    ]

    for column, (heading, description) in zip(
        workflow,
        workflow_items,
    ):

        with column:

            with st.container(border=True):
                st.markdown(f"**{heading}**")
                st.write(description)

    # --------------------------------------------------------
    # DATASET ANALYTICS GRAPHS
    # --------------------------------------------------------

    st.divider()

    st.subheader("Dataset Analytics")

    # GRAPH 1: FRAUD PREVALENCE

    st.markdown("#### 1. Fraud prevalence by dataset")

    prevalence_data = pd.DataFrame({
        "Dataset": list(DATASETS.keys()),
        "Fraud rate (%)": [
            item["fraud_rate"]
            for item in DATASETS.values()
        ],
    }).set_index("Dataset")

    st.bar_chart(
        prevalence_data,
        y="Fraud rate (%)",
        use_container_width=True,
    )

    st.caption(
        "Percentage of transactions labelled as fraudulent "
        "in each dataset."
    )

    graph_col1, graph_col2 = st.columns(2)

    # GRAPH 2: FRAUD COUNTS

    with graph_col1:

        st.markdown("#### 2. Fraud transaction counts")

        fraud_counts = pd.DataFrame({
            "Fraud cases": [
                item["fraud"]
                for item in DATASETS.values()
            ],
        }, index=list(DATASETS.keys()))

        st.bar_chart(
            fraud_counts,
            y="Fraud cases",
            use_container_width=True,
        )

        st.caption(
            "Total labelled fraud records in each dataset."
        )

    # GRAPH 3: TRANSACTION VOLUMES

    with graph_col2:

        st.markdown("#### 3. Transaction volumes")

        transaction_counts = pd.DataFrame({
            "Transactions": [
                item["transactions"]
                for item in DATASETS.values()
            ],
        }, index=list(DATASETS.keys()))

        st.bar_chart(
            transaction_counts,
            y="Transactions",
            use_container_width=True,
        )

        st.caption(
            "Total transaction records in each dataset."
        )

    # GRAPH 4: PAYSIM TEMPORAL FRAUD RATE

    st.markdown("#### 4. PaySim fraud rate over time")

    paysim_temporal = pd.DataFrame(
        PAYSIM_TEMPORAL
    ).set_index("Period")

    st.bar_chart(
        paysim_temporal,
        y="Fraud rate (%)",
        use_container_width=True,
    )

    st.caption(
        "PaySim fraud prevalence by chronological step range. "
        "Later periods show a substantial increase in fraud rate."
    )

    st.divider()

    st.info(
        "This is a research dashboard using historical and synthetic data. "
        "It is not connected to live financial transactions."
    )


# ============================================================
# 9. DATASET EXPLORER
# ============================================================

elif page == "Datasets":

    st.title("Dataset Explorer")

    st.write(
        "Explore the four datasets used in this project. "
        "Filter the view by financial domain."
    )

    domain_filter = st.multiselect(
        "Financial domain",
        ["Mobile money", "Card payments"],
        default=["Mobile money", "Card payments"],
    )

    filtered = {
        name: item
        for name, item in DATASETS.items()
        if item["domain"] in domain_filter
    }

    if not filtered:

        st.warning("Select at least one financial domain.")

    else:

        table = pd.DataFrame([
            {
                "Dataset": name,
                "Domain": item["domain"],
                "Transactions": item["transactions"],
                "Fraud cases": item["fraud"],
                "Fraud rate (%)": item["fraud_rate"],
                "Status": item["status"],
            }
            for name, item in filtered.items()
        ])

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        selected_name = st.selectbox(
            "Choose a dataset",
            list(filtered.keys()),
        )

        selected = filtered[selected_name]

        st.divider()
        st.subheader(selected_name)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Transactions",
            format_number(selected["transactions"]),
        )

        col2.metric(
            "Fraud cases",
            format_number(selected["fraud"]),
        )

        col3.metric(
            "Fraud rate",
            f"{selected['fraud_rate']:.4f}%",
        )

        st.subheader("Description")
        st.write(selected["description"])

        st.write(f"**Domain:** {selected['domain']}")
        st.write(f"**Project status:** {selected['status']}")

        st.subheader("Known limitation")
        st.warning(selected["limitation"])

        st.caption(
            "Statistics are based on the project's dataset audits."
        )


# ============================================================
# 10. DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.title("Data Quality and Risk Analysis")

    st.write(
        "Data quality describes how complete, consistent and suitable "
        "data is for analysis and machine learning."
    )

    quality_data = pd.DataFrame([
        {
            "Dataset": "MoMTSim V1",
            "Finding": "Fraud labels concentrated in TRANSFER",
            "Research concern": "Synthetic label patterns",
        },
        {
            "Dataset": "MoMTSim V2",
            "Finding": "Fraud prevalence approximately 52.84%",
            "Research concern": "Generator-specific patterns",
        },
        {
            "Dataset": "PaySim",
            "Finding": "Fraud prevalence changes over time",
            "Research concern": "Temporal distribution shift",
        },
        {
            "Dataset": "IEEE-CIS",
            "Finding": "Many missing identity features",
            "Research concern": "Missingness and generalization",
        },
    ])

    st.subheader("Dataset audit findings")

    st.dataframe(
        quality_data,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    definitions = {
        "Missing values": (
            "Information that is not recorded for a particular transaction. "
            "Missingness may be normal in some systems and should not "
            "automatically be treated as an error."
        ),
        "Class imbalance": (
            "A situation where one category is much more common than "
            "another. Fraud datasets often contain many more legitimate "
            "transactions than fraudulent ones."
        ),
        "Data leakage": (
            "Using information during training that would not legitimately "
            "be available when making a prediction. Leakage can produce "
            "misleadingly optimistic evaluation results."
        ),
        "Temporal distribution shift": (
            "A change in transaction behaviour or fraud prevalence over "
            "time. A model trained on older records may perform differently "
            "on newer records."
        ),
        "Synthetic-data artifacts": (
            "Patterns introduced by the software that generated synthetic "
            "data. Models may learn these artificial rules instead of "
            "patterns that occur in real financial systems."
        ),
        "Feature availability": (
            "A feature is appropriate for prediction only if its information "
            "would be available at the time the prediction is made."
        ),
    }

    st.subheader("Data quality concepts")

    for title, description in definitions.items():

        with st.expander(title):
            st.write(description)

    st.divider()

    st.subheader("Feature selection policy")

    st.write(
        "Post-transaction balance fields and variables derived from "
        "post-transaction information require special caution. They may "
        "not be available when a pre-authorization decision is made."
    )

    st.write(
        "The PaySim feature pipeline excludes `isFlaggedFraud`, "
        "which represents an existing detection flag rather than "
        "an independent transaction input."
    )

    st.write(
        "The IEEE-CIS pipeline joins selected identity features "
        "to transaction records using TransactionID."
    )


# ============================================================
# 11. MODEL LAB
# ============================================================

elif page == "Model Lab":

    st.title("Model Lab")

    st.write(
        "A classification model learns patterns from labelled historical "
        "transactions and produces scores that can be used to rank "
        "transactions by estimated fraud likelihood."
    )

    selected_model = st.selectbox(
        "Select a trained model",
        list(MODEL_PATHS.keys()),
    )

    result = load_evaluation(
        str(MODEL_PATHS[selected_model])
    )

    artifact_path = MODEL_ARTIFACTS[selected_model]

    if result is None:

        st.error("The model evaluation file could not be found.")

    else:

        validation = result.get("validation", {})
        test = result.get("test", {})

        st.subheader(f"{selected_model} baseline results")

        threshold = result.get("threshold")

        if threshold is not None:

            st.metric(
                "Validation-selected threshold",
                f"{threshold:.6f}",
            )

        st.caption(
            "The threshold is selected using validation data and then "
            "applied unchanged to the test data."
        )

        metric_keys = [
            "average_precision",
            "roc_auc",
            "precision",
            "recall",
            "f1",
        ]

        metric_names = [
            "PR-AUC",
            "ROC-AUC",
            "Precision",
            "Recall",
            "F1",
        ]

        metric_columns = st.columns(5)

        for column, key, label in zip(
            metric_columns,
            metric_keys,
            metric_names,
        ):

            value = test.get(key)

            with column:

                st.metric(
                    label,
                    f"{value:.4f}" if value is not None else "N/A",
                    help=metric_definition(label),
                )

        st.divider()

        # VALIDATION VERSUS TEST GRAPH

        st.subheader("Validation versus test performance")

        comparison_data = {}

        for label, values in [
            ("Validation", validation),
            ("Test", test),
        ]:

            comparison_data[label] = {
                "PR-AUC": values.get("average_precision"),
                "ROC-AUC": values.get("roc_auc"),
                "Precision": values.get("precision"),
                "Recall": values.get("recall"),
                "F1": values.get("f1"),
            }

        comparison = pd.DataFrame(comparison_data)

        st.bar_chart(
            comparison,
            use_container_width=True,
        )

        st.caption(
            "Validation data is used for model selection. Test data is "
            "held aside for chronological evaluation."
        )

        st.divider()

        # CONFUSION MATRIX

        st.subheader("Confusion matrix")

        matrix = test.get("confusion_matrix")

        if matrix is not None:

            confusion = pd.DataFrame(
                matrix,
                index=[
                    "Actual legitimate",
                    "Actual fraud",
                ],
                columns=[
                    "Predicted legitimate",
                    "Predicted fraud",
                ],
            )

            st.dataframe(
                confusion,
                use_container_width=True,
            )

            st.write(
                "The confusion matrix shows correct and incorrect "
                "predictions for both classes."
            )

            st.markdown("#### Confusion matrix visualization")

            confusion_chart = pd.DataFrame({
                "Correct prediction": [
                    matrix[0][0],
                    matrix[1][1],
                ],
                "Incorrect prediction": [
                    matrix[0][1],
                    matrix[1][0],
                ],
            }, index=["Legitimate", "Fraud"])

            st.bar_chart(
                confusion_chart,
                use_container_width=True,
            )

        # INVESTIGATION CAPACITY

        if (
            test.get("precision_at_1_percent") is not None
            and test.get("recall_at_1_percent") is not None
        ):

            st.divider()
            st.subheader("Investigation capacity")

            col1, col2 = st.columns(2)

            col1.metric(
                "Precision at 1% review capacity",
                f"{test['precision_at_1_percent']:.4f}",
            )

            col2.metric(
                "Recall at 1% review capacity",
                f"{test['recall_at_1_percent']:.4f}",
            )

            st.caption(
                "These values evaluate the highest-scoring 1% of test "
                "transactions as a potential investigation queue."
            )

        st.divider()

        st.subheader("Saved model artifact")

        if artifact_path.exists():
            st.success("The trained model file is present.")
        else:
            st.warning("The trained model file was not found.")

        if selected_model == "IEEE-CIS":

            st.warning(
                "IEEE-CIS test performance is weaker than validation. "
                "Its test PR-AUC is approximately 0.0836 and ROC-AUC "
                "approximately 0.6352. Temporal generalization needs "
                "further investigation."
            )

        elif selected_model == "PaySim":

            st.info(
                "PaySim has substantial changes in fraud prevalence "
                "over time. Its chronological test results should be "
                "interpreted as an out-of-time stress test."
            )

        st.caption(
            "All displayed scores come from saved evaluation results. "
            "They are historical results, not predictions for live transactions."
        )


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

elif page == "Comparison":

    st.title("Model Comparison")

    st.write(
        "Compare the saved PaySim and IEEE-CIS baseline evaluation results."
    )

    st.warning(
        "These models were trained on different datasets with different "
        "fraud prevalence and feature distributions. This is a descriptive "
        "comparison, not a controlled head-to-head experiment."
    )

    results = get_model_results()

    if results.empty:

        st.warning("No model evaluation files were found.")

    else:

        st.subheader("Test performance")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

        selected_metric = st.selectbox(
            "Choose a metric",
            [
                "PR-AUC",
                "ROC-AUC",
                "Precision",
                "Recall",
                "F1",
            ],
        )

        chart_data = results.set_index("Dataset")[
            [selected_metric]
        ]

        st.subheader(f"{selected_metric} comparison")

        st.bar_chart(
            chart_data,
            use_container_width=True,
        )

        st.write(metric_definition(selected_metric))

        st.divider()

        st.subheader("All model metrics")

        all_metrics = results.set_index("Dataset")[
            [
                "PR-AUC",
                "ROC-AUC",
                "Precision",
                "Recall",
                "F1",
            ]
        ]

        st.line_chart(
            all_metrics,
            use_container_width=True,
        )

        st.caption(
            "Interpret each score in the context of its dataset, "
            "fraud prevalence, chronological split and feature set."
        )


# ============================================================
# 13. RESEARCH REPORTS
# ============================================================

elif page == "Reports":

    st.title("Research Report Library")

    st.write(
        "Browse reports generated during dataset auditing, feature "
        "engineering, model evaluation and provenance research."
    )

    if not REPORT_DIR.exists():

        st.error("The reports directory was not found.")

    else:

        reports = sorted(REPORT_DIR.glob("*.md"))

        if not reports:

            st.info("No Markdown reports were found.")

        else:

            search = st.text_input(
                "Search report titles",
                placeholder="Search reports...",
            )

            filtered_reports = [
                report
                for report in reports
                if search.lower() in report.stem.lower()
            ]

            if not filtered_reports:

                st.info("No reports match your search.")

            else:

                selected_report = st.selectbox(
                    "Select a report",
                    filtered_reports,
                    format_func=lambda path: (
                        path.stem.replace("_", " ").title()
                    ),
                )

                st.divider()

                report_text = selected_report.read_text(
                    encoding="utf-8"
                )

                st.markdown(report_text)


# ============================================================
# 14. GLOSSARY
# ============================================================

elif page == "Glossary":

    st.title("Financial Fraud Detection Glossary")

    st.write(
        "A plain-language dictionary of technical terms used "
        "throughout this project."
    )

    glossary = {
        "Artificial intelligence": (
            "Computer systems performing tasks that normally require "
            "human-like reasoning, learning or pattern recognition."
        ),
        "Machine learning": (
            "A method in which a computer learns patterns from examples "
            "instead of relying only on manually written rules."
        ),
        "Financial fraud": (
            "An intentional attempt to obtain money or financial benefits "
            "through deception, misuse or unauthorized activity."
        ),
        "Transaction": (
            "An event involving the movement, payment or exchange of money."
        ),
        "Dataset": (
            "An organized collection of records used for analysis or training."
        ),
        "Feature": (
            "An input variable used by a machine-learning model, such as "
            "transaction amount or transaction type."
        ),
        "Target variable": (
            "The outcome a model is trained to predict. Here it represents "
            "whether a transaction is labelled as fraud."
        ),
        "Label": (
            "The known answer attached to a training example, such as "
            "fraudulent or legitimate."
        ),
        "Class imbalance": (
            "A situation where one class is much more common than another."
        ),
        "Preprocessing": (
            "Preparing raw data for a model, including handling missing "
            "values, scaling numbers and encoding categories."
        ),
        "Imputation": (
            "Replacing missing data with a chosen value, such as a median "
            "or a special category."
        ),
        "Standardization": (
            "Rescaling numerical features so that they have a comparable scale."
        ),
        "One-hot encoding": (
            "Converting categories into separate binary columns so that "
            "a machine-learning algorithm can process them."
        ),
        "Training set": (
            "The portion of data used to teach a model."
        ),
        "Validation set": (
            "A separate portion used to compare model settings and "
            "select a classification threshold."
        ),
        "Test set": (
            "Data held aside to estimate how well a model performs "
            "on unseen examples."
        ),
        "Chronological split": (
            "Dividing records according to time, generally using older "
            "records for training and newer records for testing."
        ),
        "Data leakage": (
            "Using information during training that would not legitimately "
            "be available when making a prediction."
        ),
        "Feature leakage": (
            "A feature reveals the target or uses information unavailable "
            "at the intended prediction time."
        ),
        "Logistic regression": (
            "A classification algorithm that estimates the probability "
            "of an example belonging to a particular class."
        ),
        "SGDClassifier": (
            "A scikit-learn classifier trained using stochastic gradient "
            "descent, an optimization method that updates model parameters."
        ),
        "Classification": (
            "Assigning an example to a category, such as fraud or legitimate."
        ),
        "Probability score": (
            "A numerical model output representing its estimated likelihood "
            "that an example belongs to a class."
        ),
        "Threshold": (
            "The score cutoff used to decide whether a transaction "
            "is flagged as fraud."
        ),
        "True positive": (
            "A fraudulent transaction correctly identified as fraud."
        ),
        "False positive": (
            "A legitimate transaction incorrectly flagged as fraud."
        ),
        "True negative": (
            "A legitimate transaction correctly classified as legitimate."
        ),
        "False negative": (
            "A fraudulent transaction incorrectly classified as legitimate."
        ),
        "Precision": (
            "Among transactions flagged as fraud, the proportion "
            "that really are fraudulent."
        ),
        "Recall": (
            "Among actual fraud cases, the proportion detected by the model."
        ),
        "F1 score": (
            "A combined score that balances precision and recall."
        ),
        "ROC-AUC": (
            "A measure of how well a model ranks positive examples above "
            "negative examples across classification thresholds."
        ),
        "PR-AUC": (
            "The area under the precision-recall curve. It is especially "
            "useful when the positive class, such as fraud, is rare."
        ),
        "Confusion matrix": (
            "A table showing true positives, false positives, true negatives "
            "and false negatives."
        ),
        "Class weight": (
            "A training adjustment that gives more importance to a "
            "less common class."
        ),
        "Temporal distribution shift": (
            "A change in transaction characteristics or fraud prevalence "
            "between earlier and later periods."
        ),
        "Synthetic data": (
            "Artificially generated data designed to resemble records "
            "from a real-world process."
        ),
        "Model generalization": (
            "The ability of a model to perform well on new data it "
            "did not see during training."
        ),
        "Out-of-time testing": (
            "Testing on later transactions than those used for training "
            "and validation."
        ),
        "Review capacity": (
            "The number or percentage of transactions investigators "
            "can manually inspect."
        ),
        "Average precision": (
            "A summary of precision across recall levels, commonly used "
            "as a numerical summary of the precision-recall curve."
        ),
        "Fraud prevalence": (
            "The percentage of records in a dataset labelled as fraudulent."
        ),
        "False alarm": (
            "A legitimate transaction incorrectly identified as suspicious."
        ),
        "Baseline model": (
            "An initial model used as a reference point for evaluating "
            "whether later models provide improvements."
        ),
        "Hyperparameter": (
            "A model setting selected before or during training, such as "
            "regularization strength or number of iterations."
        ),
        "Regularization": (
            "A method that discourages overly complex model parameters "
            "to help reduce overfitting."
        ),
        "Overfitting": (
            "When a model learns details specific to training data and "
            "performs poorly on new data."
        ),
    }

    search_term = st.text_input(
        "Search glossary",
        placeholder="Try precision, recall, threshold, leakage...",
    )

    filtered_glossary = {
        term: definition
        for term, definition in glossary.items()
        if search_term.lower() in term.lower()
        or search_term.lower() in definition.lower()
    }

    st.caption(f"{len(filtered_glossary)} definitions found")

    for term, definition in filtered_glossary.items():

        with st.expander(term):
            st.write(definition)


# ============================================================
# 15. ABOUT
# ============================================================

elif page == "About":

    st.title("About This Project")

    st.subheader("Project purpose")

    st.write(
        "The purpose of this project is to investigate how machine "
        "learning can help identify potentially fraudulent financial "
        "transactions across different financial transaction environments."
    )

    st.write(
        "It combines dataset auditing, temporal analysis, feature "
        "engineering, baseline model training and performance evaluation."
    )

    st.divider()

    st.subheader("Datasets used")

    st.markdown("""
    **MoMTSim V1 and V2**

    Synthetic mobile-money transaction datasets used to investigate
    transaction patterns, fraud labels and simulator-specific behaviour.

    **PaySim**

    A simulated mobile-money dataset used for chronological fraud
    detection experiments.

    **IEEE-CIS**

    A card transaction fraud dataset containing transaction and
    identity information.
    """)

    st.divider()

    st.subheader("Project methodology")

    methodology = [
        "Inspect dataset structure, transaction counts and labels.",
        "Audit fraud prevalence and transaction characteristics.",
        "Investigate temporal patterns and potential data leakage.",
        "Select features appropriate for the prediction task.",
        "Create chronological training, validation and test partitions.",
        "Train baseline machine-learning classifiers.",
        "Evaluate with PR-AUC, ROC-AUC, precision, recall and F1.",
        "Document limitations and compare experimental results.",
    ]

    for index, step in enumerate(methodology, start=1):
        st.markdown(f"**{index}.** {step}")

    st.divider()

    st.subheader("Technology stack")

    st.write(
        "Python, Pandas, NumPy, Scikit-learn, Joblib and Streamlit."
    )

    st.divider()

    st.subheader("Research limitations")

    st.warning(
        "The datasets include synthetic and simulated transaction data. "
        "Performance on these datasets does not establish performance "
        "on real financial transactions."
    )

    st.write(
        "PaySim and IEEE-CIS are separate models trained on different "
        "data sources. Their results are not a controlled head-to-head "
        "comparison."
    )

    st.write(
        "MoMTSim has analysis and evaluation reports, but a saved "
        "MoMTSim model artifact is not currently registered in this app."
    )

    st.write(
        "This application does not connect to banking infrastructure, "
        "process live payments or make actual financial decisions."
    )


# ============================================================
# 16. USER MANAGEMENT
# ============================================================

elif page == "User Management":

    show_user_management()


# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="app-footer">FRAUD INTELLIGENCE</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Multi-Channel Financial Fraud Detection Research Project"
)

st.caption(
    "Educational and research use only. "
    "Not a production fraud detection system."
)
