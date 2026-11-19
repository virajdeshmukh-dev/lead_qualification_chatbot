import tomllib
from pathlib import Path

from lead_qualification_chatbot.config import DEFAULT_DATA_DIR, Settings


def test_console_script_uses_the_real_cli() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["lead-qualification-chatbot"] == (
        "lead_qualification_chatbot.cli:main"
    )


def test_default_data_directory_is_packaged_with_the_module() -> None:
    cfg = Settings(_env_file=None, llm_provider="ollama", llm_model="test-model")

    assert cfg.data_dir == DEFAULT_DATA_DIR
    assert {path.name for path in DEFAULT_DATA_DIR.glob("*.csv")} == {
        "faq.csv",
        "properties_for_buyers.csv",
        "seller_coverage.csv",
    }


def test_local_secrets_and_vector_state_are_ignored() -> None:
    project_root = Path(__file__).resolve().parents[2]
    ignored = set((project_root / ".gitignore").read_text().splitlines())

    assert {".env", ".chroma/"} <= ignored
