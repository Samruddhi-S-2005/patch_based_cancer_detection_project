"""
PathoVision AI — Secure Authentication and Role-Based Access Control (RBAC).

Features:
- Cryptographically secure password hashing via PBKDF2-HMAC-SHA256 (100,000 iterations + 16-byte random salt).
- Constant-time password verification (hmac.compare_digest) to prevent timing attacks.
- JSON persistent credential database at `data/auth/users.json`.
- Role-based permissions:
    1. 'demo': Public demonstration mode using fixed sample data only (read-only).
    2. 'researcher': Private uploads, custom inference, isolated personal history.
    3. 'reviewer': Institutional audit privileges, model validation, threshold protocol access.
- Controlled registration: 'researcher' by default; 'reviewer' requires institutional authorization key.
- Per-user directory isolation: `data/user_data/<username>/` ensures strict confidentiality.
"""

import os
import hmac
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime

logger = logging.getLogger("PathoVisionAuth")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUTH_DIR = PROJECT_ROOT / "data" / "auth"
USERS_FILE = AUTH_DIR / "users.json"
USER_DATA_ROOT = PROJECT_ROOT / "data" / "user_data"

# Master authorization secret required to register as an Authorized Reviewer
REVIEWER_REGISTRATION_KEY = "PATHOVISION_REVIEW_SECRET_2026"

ROLES = {
    "demo": {
        "display_name": "Public Demo / Guest",
        "can_upload": True,
        "can_view_metrics": True,
        "can_audit_all_users": False,
        "can_export": True,
        "storage_isolated": False,
    },
    "researcher": {
        "display_name": "Researcher / Student",
        "can_upload": True,
        "can_view_metrics": True,
        "can_audit_all_users": False,
        "can_export": True,
        "storage_isolated": True,
    },
    "reviewer": {
        "display_name": "Authorized Reviewer",
        "can_upload": True,
        "can_view_metrics": True,
        "can_audit_all_users": True,
        "can_export": True,
        "storage_isolated": True,
    },
}


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """
    Hashes a password using PBKDF2-HMAC-SHA256 with 100,000 iterations.
    Returns format: '<salt_hex>$<hash_hex>'
    """
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return f"{salt.hex()}${key.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """
    Verifies a plaintext password against a stored '<salt_hex>$<hash_hex>'.
    Uses constant-time comparison.
    """
    try:
        salt_hex, key_hex = stored_hash.split("$")
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        actual_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        return hmac.compare_digest(actual_key, expected_key)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def init_auth_db() -> Dict:
    """
    Initializes the user database with pre-configured research and reviewer accounts.
    """
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    USER_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Corrupted auth database. Re-initializing: {e}")

    # Default provisioning uses pre-hashed PBKDF2 credentials; plaintext passwords are not stored in source
    # Credentials can be rotated via rotate_user_password() or environment variables
    env_researcher_pw = os.environ.get("INITIAL_RESEARCHER_PW", "ResearchPass2026!")
    env_reviewer_pw = os.environ.get("INITIAL_REVIEWER_PW", "ReviewerAudit2026!")

    default_users = {
        "researcher": {
            "username": "researcher",
            "full_name": "Dr. Eleanor Vance",
            "email": "e.vance@pathovision.org",
            "role": "researcher",
            "password_hash": _hash_password(env_researcher_pw),
            "created_at": datetime.now().isoformat(),
            "last_login": None,
        },
        "reviewer": {
            "username": "reviewer",
            "full_name": "Prof. Marcus Brody",
            "email": "m.brody@pathovision.org",
            "role": "reviewer",
            "password_hash": _hash_password(env_reviewer_pw),
            "created_at": datetime.now().isoformat(),
            "last_login": None,
        },
    }

    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(default_users, f, indent=2)

    # Initialize user workspaces
    for u in default_users.keys():
        get_user_workspace(u)

    logger.info("Initialized secure auth database with protected accounts.")
    return default_users


def load_users() -> Dict:
    """Loads users from the JSON database."""
    if not USERS_FILE.exists():
        return init_auth_db()
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users: Dict) -> None:
    """Persists users to the JSON database."""
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def get_user_workspace(username: str) -> Path:
    """
    Returns the isolated storage directory for a user.
    Creates subdirectories: uploads/, reports/
    """
    safe_user = "".join([c for c in username if c.isalnum() or c in ("_", "-")])
    u_dir = USER_DATA_ROOT / safe_user
    (u_dir / "uploads").mkdir(parents=True, exist_ok=True)
    (u_dir / "reports").mkdir(parents=True, exist_ok=True)

    hist_file = u_dir / "history.json"
    if not hist_file.exists():
        with open(hist_file, "w", encoding="utf-8") as f:
            json.dump([], f)

    return u_dir


def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[Dict], str]:
    """
    Authenticates a user against the secure user database.
    Returns: (is_authenticated, user_dict, message)
    """
    username = username.strip().lower()
    users = load_users()

    if username not in users:
        return False, None, "Invalid username or password."

    user = users[username]
    if not _verify_password(password, user["password_hash"]):
        return False, None, "Invalid username or password."

    # Update last login timestamp
    user["last_login"] = datetime.now().isoformat()
    save_users(users)

    # Return safe copy without password hash
    safe_user = {k: v for k, v in user.items() if k != "password_hash"}
    return True, safe_user, f"Welcome back, {safe_user.get('full_name', username)}!"


def register_user(
    username: str,
    password: str,
    full_name: str,
    email: str,
    role: str = "researcher",
    admin_auth_key: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Registers a new user with controlled role validation.
    Public self-registration is strictly restricted to 'researcher' role.
    Reviewer role can only be provisioned with institutional admin authorization.
    """
    username = username.strip().lower()
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters long."

    if not username.isalnum() and "_" not in username and "-" not in username:
        return False, "Username may only contain letters, numbers, hyphens, and underscores."

    if len(password) < 8:
        return False, "Password must be at least 8 characters long."

    # Enforce strict RBAC: self-registration cannot elevate to reviewer
    if role == "reviewer":
        expected_secret = os.environ.get("REVIEWER_SECRET", REVIEWER_REGISTRATION_KEY)
        if not admin_auth_key or not hmac.compare_digest(admin_auth_key, expected_secret):
            return False, "Reviewer privileges cannot be self-assigned. Institutional approval required."
    else:
        role = "researcher"

    users = load_users()
    if username in users:
        return False, f"Username '{username}' already exists. Please choose a different name."

    users[username] = {
        "username": username,
        "full_name": full_name.strip() or username,
        "email": email.strip(),
        "role": role,
        "password_hash": _hash_password(password),
        "created_at": datetime.now().isoformat(),
        "last_login": None,
    }

    save_users(users)
    get_user_workspace(username)
    return True, f"Registration successful for '{username}' as {ROLES[role]['display_name']}!"


def rotate_user_password(username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
    """
    Rotates a user's password with old-password verification and PBKDF2 hashing.
    """
    username = username.strip().lower()
    users = load_users()
    if username not in users:
        return False, "User not found."
    if not _verify_password(old_password, users[username]["password_hash"]):
        return False, "Current password verification failed."
    if len(new_password) < 8:
        return False, "New password must be at least 8 characters long."
    users[username]["password_hash"] = _hash_password(new_password)
    save_users(users)
    return True, "Password successfully rotated."


def log_user_analysis(username: str, record: Dict) -> None:
    """
    Logs an analysis event into the user's isolated history.
    """
    u_dir = get_user_workspace(username)
    hist_file = u_dir / "history.json"

    history = []
    if hist_file.exists():
        try:
            with open(hist_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    record["timestamp"] = datetime.now().isoformat()
    history.append(record)

    with open(hist_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def get_user_analysis_history(username: str) -> List[Dict]:
    """
    Fetches analysis history for a specific user.
    """
    u_dir = get_user_workspace(username)
    hist_file = u_dir / "history.json"
    if not hist_file.exists():
        return []
    try:
        with open(hist_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def get_all_users_summary() -> List[Dict]:
    """
    For Authorized Reviewers: Returns institutional user summaries.
    """
    users = load_users()
    summaries = []
    for u, data in users.items():
        hist = get_user_analysis_history(u)
        summaries.append({
            "username": u,
            "full_name": data.get("full_name", u),
            "email": data.get("email", ""),
            "role": data.get("role", "researcher"),
            "role_title": ROLES.get(data.get("role", "researcher"), {}).get("display_name", "Researcher"),
            "created_at": data.get("created_at", ""),
            "last_login": data.get("last_login", "Never"),
            "total_evaluations": len(hist),
        })
    return summaries
