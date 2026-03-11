"""
app/config.py — Central configuration loaded from .env
Paths are computed as ABSOLUTE values anchored to the project root so
the same vault.db is used regardless of the working directory Streamlit
or any other entry-point is launched from.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of the `app/` package directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Resolve .env from the project root so it works from any cwd
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """
    Central configuration class for the eshu application.
    Loaded automatically from the local `.env` file using Pydantic.
    """
    
    # ─── Ollama ────────────────────────────────────────────
    OLLAMA_HOST: str = "http://localhost:11434"
    MAIN_MODEL: str = "llama3.1"
    ROUTER_MODEL: str = "phi3"
    EMBED_MODEL: str = "nomic-embed-text"
    VISION_MODEL: str = "llava"
    ROUTER_TIMEOUT: int = 180  # seconds for Phi routing (first load can be slow)

    # ─── Chunking ──────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ─── Retrieval ─────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 5
    EMBED_TIMEOUT: int = 180  # seconds for Ollama embedding API (first load can be slow)

    # ─── Router Thresholds ─────────────────────────────────
    ROUTER_CONFIDENCE_HIGH: float = 0.85
    ROUTER_CONFIDENCE_MED: float = 0.60
    MULTI_VAULT_THRESHOLD: float = 0.35

    # ─── Learning System ───────────────────────────────────
    CONTEXT_WINDOW_TURNS: int = 5
    RULE_EXTRACTION_THRESHOLD: int = 3

    # ─── Paths (relative values from .env are made absolute below) ──
    DATA_DIR: str = "data"
    SQLITE_PATH: str = ""          # computed in __init__ if blank

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="allow"
    )

    def model_post_init(self, __context):
        """
        Pydantic hook called after model initialization.
        Resolves all relative path configurations (like DATA_DIR and SQLITE_PATH) 
        to absolute paths anchored to this file's parent directory (the project root).
        This guarantees that the app always interacts with the correct database and 
        data folder regardless of the current working directory of the terminal where 
        the server or Streamlit app was launched.
        """
        # DATA_DIR → absolute
        d = Path(self.DATA_DIR)
        if not d.is_absolute():
            d = _PROJECT_ROOT / d
        object.__setattr__(self, "DATA_DIR", str(d))

        # SQLITE_PATH → absolute (default: DATA_DIR/vault.db)
        if not self.SQLITE_PATH:
            object.__setattr__(self, "SQLITE_PATH", str(d / "vault.db"))
        else:
            sp = Path(self.SQLITE_PATH)
            if not sp.is_absolute():
                sp = _PROJECT_ROOT / sp
            object.__setattr__(self, "SQLITE_PATH", str(sp))

    @property
    def data_path(self) -> Path:
        """Returns the absolute Path to the main data directory, creating it if necessary."""
        p = Path(self.DATA_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def sqlite_path(self) -> Path:
        """Returns the absolute Path to the SQLite database file."""
        return Path(self.SQLITE_PATH)

    @property
    def registry_path(self) -> Path:
        """Returns the absolute Path to the vault registry JSON file."""
        return self.data_path / "registry.json"

    @property
    def rules_path(self) -> Path:
        """Returns the absolute Path to the user routing rules JSON file."""
        return self.data_path / "user_rules.json"

    def vault_dir(self, vault_name: str) -> Path:
        """
        Returns the absolute Path to the specific directory for a given vault,
        creating the directory automatically if it does not exist.
        """
        p = self.data_path / "vaults" / vault_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def fingerprint_path(self, vault_name: str) -> Path:
        """Returns the path to the fingerprint profile JSON file for a given vault."""
        return self.vault_dir(vault_name) / "fingerprint.json"

    def meta_path(self, vault_name: str) -> Path:
        """Returns the path to the dynamic metadata JSON file for a given vault."""
        return self.vault_dir(vault_name) / "meta.json"


# Global singleton settings instance used by all submodules
settings = Settings()
