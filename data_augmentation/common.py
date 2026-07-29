"""Utilitários compartilhados pelos scripts de aumento de dados (data augmentation).

Centraliza a localização das pastas de entrada/saída, o carregamento de
variáveis de ambiente (.env) e a configuração de logging, evitando a
duplicação que existia entre generate_database_albumentations.py,
generate_database_augraphy.py e generate_database_openai.py.
"""

import logging
import os
import sys
from pathlib import Path

# Resolvidos a partir da localização deste arquivo, não do diretório de onde
# o script é chamado (mais robusto que o antigo Path("../database")).
BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR.parent / "database"
INPUT_DIR = DATABASE_DIR / "input"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_env_file(path: Path = BASE_DIR / ".env") -> None:
    """Carrega variáveis de um arquivo .env simples (KEY=VALUE) para o
    ambiente do processo, sem exigir a dependência python-dotenv.
    Não sobrescreve variáveis já definidas no ambiente.
    """
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def get_input_files() -> list[Path]:
    """Retorna as imagens válidas em INPUT_DIR, validando a pasta antes."""
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de entrada não encontrada: {INPUT_DIR}. "
            "Crie a pasta e adicione as imagens antes de rodar o script."
        )

    files = sorted(
        f for f in INPUT_DIR.iterdir() if f.suffix.lower() in VALID_EXTENSIONS
    )

    if not files:
        raise FileNotFoundError(
            f"Nenhuma imagem (.jpg/.jpeg/.png) encontrada em {INPUT_DIR}."
        )

    return files


def ensure_output_dir(output_dir: Path) -> Path:
    """Garante que a pasta de saída exista (cria pastas pai se necessário)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def setup_logger(name: str, output_dir: Path) -> logging.Logger:
    """Configura um logger que grava no console e em '<output_dir>/<name>.log'."""
    ensure_output_dir(output_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(output_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
