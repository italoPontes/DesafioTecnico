import base64
import os
import sys
import time

from itertools import product

from openai import OpenAI

from common import (
    BASE_DIR,
    DATABASE_DIR,
    ensure_output_dir,
    get_input_files,
    load_env_file,
    setup_logger,
)

OUTPUT_DIR = ensure_output_dir(DATABASE_DIR / "openai")
logger = setup_logger("openai", OUTPUT_DIR)

load_env_file()

configurations = [
    {
        "name": "darken",
        "prompt": "Simulate that this image was captured in a very dark environment, but the image content is still visible."
    },
    {
        "name": "brighten",
        "prompt": "Simulate that this image was captured in a very bright environment, but the image content is still visible."
    },
    {
        "name": "low_quality_camera",
        "prompt": "Simulate that this same photo was captured using a low-quality camera."
    },
    {
        "name": "glare",
        "prompt": "Simulate a strong light glare in this image, as if the camera lens were dirty or reflecting intense light, while the image content remains visible."
    },
    {
        "name": "crop",
        "prompt": "Simulate that the image has been cropped, removing parts of the image and making the content harder to view."
    }
]

logger.info(f"Number of modifications: {len(configurations)}")

files = get_input_files()
logger.info(f"Files found: {len(files)}")


def load_api_key() -> str:
    """
    Essa função carrega a chave da OpenAI a partir da variável de ambiente
    OPENAI_API_KEY ou do arquivo openai.txt (depreciado).
    Se não encontrar a chave, lança um RuntimeError.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    legacy_path = BASE_DIR / "openai.txt"
    if legacy_path.exists():
        content = legacy_path.read_text().strip()
        if content and not content.startswith("#"):
            logger.warning(
                "Lendo a chave a partir de openai.txt (depreciado e inseguro). "
                "Defina OPENAI_API_KEY em data_augmentation/.env."
            )
            return content

    raise RuntimeError(
        "Chave da OpenAI não encontrada. Defina OPENAI_API_KEY em "
        "data_augmentation/.env (veja .env.example)."
    )


client = OpenAI(api_key=load_api_key())

MAX_RETRIES = 3


def edit_image(input_image: str, prompt: str) -> bytes:
    """
    Função para editar uma imagem usando a API da OpenAI.
    Recebe o caminho da imagem de entrada e o prompt de edição.
    Retorna os bytes da imagem editada.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(input_image, "rb") as image_file:
                result = client.images.edit(
                    model="gpt-image-2",
                    image=image_file,
                    prompt=prompt,
                )
            return base64.b64decode(result.data[0].b64_json)
        except Exception as exc:
            last_error = exc
            logger.warning(
                f"Tentativa {attempt}/{MAX_RETRIES} falhou para {input_image}: {exc}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Falha ao editar {input_image} após {MAX_RETRIES} tentativas"
    ) from last_error


def save_image(image_bytes: bytes, output_file_name) -> None:
    """
    Função para salvar os bytes da imagem editada em um arquivo.
    Recebe os bytes da imagem e o caminho de saída.
    """
    with open(output_file_name, "wb") as file:
        file.write(image_bytes)
    logger.info(f"Image saved to: {output_file_name}")


def process(image, configuration) -> None:
    """
    Função principal que processa cada imagem com cada configuração.
    Recebe a imagem e a configuração (nome e prompt), chama
    edit_image e save_image.
    """
    logger.info(f"{image.name} -> {configuration['name']}")

    output_file_name = (
        OUTPUT_DIR / f"{image.stem}_{configuration['name']}{image.suffix}"
    )

    try:
        output_image = edit_image(
            input_image=str(image),
            prompt=configuration["prompt"]
        )

        save_image(
            image_bytes=output_image,
            output_file_name=output_file_name
        )
    except Exception as exc:
        logger.error(f"Failed on {image.name} -> {configuration['name']}: {exc}")


total_calls = len(files) * len(configurations)
logger.warning(
    f"Esta execução fará {total_calls} chamadas pagas à API da OpenAI "
    "(gpt-image-2). Confira os preços atuais em https://openai.com/api/pricing "
    "antes de continuar."
)

if sys.stdin.isatty():
    if input("Continuar? [y/N] ").strip().lower() != "y":
        logger.info("Execução cancelada pelo usuário.")
        sys.exit(0)

for image, configuration in product(files, configurations):
    process(image, configuration)

logger.info("Finished.")
