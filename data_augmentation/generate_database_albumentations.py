import cv2
import albumentations as A

from itertools import product

from common import DATABASE_DIR, ensure_output_dir, get_input_files, setup_logger

OUTPUT_DIR = ensure_output_dir(DATABASE_DIR / "albumentations")
logger = setup_logger("albumentations", OUTPUT_DIR)

augmentations = [
    {"name": "AdvancedBlur", "action": A.Compose([A.AdvancedBlur(p=1.0)])},
    {"name": "Blur", "action": A.Compose([A.Blur(p=1.0)])},
    {"name": "GaussianBlur", "action": A.Compose([A.GaussianBlur(p=1.0)])},
    {"name": "MotionBlur", "action": A.Compose([A.MotionBlur(p=1.0)])},
    {"name": "Defocus", "action": A.Compose([A.Defocus(p=1.0)])},
    {"name": "ZoomBlur", "action": A.Compose([A.ZoomBlur(p=1.0)])},

    {"name": "GaussNoise", "action": A.Compose([A.GaussNoise(p=1.0)])},
    {"name": "ISONoise", "action": A.Compose([A.ISONoise(p=1.0)])},
    {"name": "MultiplicativeNoise", "action": A.Compose([A.MultiplicativeNoise(p=1.0)])},

    {"name": "RandomBrightnessContrast", "action": A.Compose([A.RandomBrightnessContrast(p=1.0)])},
    {"name": "RandomGamma", "action": A.Compose([A.RandomGamma(p=1.0)])},
    {"name": "ColorJitter", "action": A.Compose([A.ColorJitter(p=1.0)])},
    {"name": "HueSaturationValue", "action": A.Compose([A.HueSaturationValue(p=1.0)])},
    {"name": "CLAHE", "action": A.Compose([A.CLAHE(p=1.0)])},

    {"name": "ImageCompression", "action": A.Compose([A.ImageCompression(p=1.0)])},

    {"name": "Perspective", "action": A.Compose([A.Perspective(p=1.0)])},
    {"name": "Affine", "action": A.Compose([A.Affine(p=1.0)])},
    {"name": "ShiftScaleRotate", "action": A.Compose([A.ShiftScaleRotate(p=1.0)])},

    {"name": "RandomShadow", "action": A.Compose([A.RandomShadow(p=1.0)])},
    {"name": "RandomSunFlare", "action": A.Compose([A.RandomSunFlare(p=1.0)])},
    {"name": "RandomFog", "action": A.Compose([A.RandomFog(p=1.0)])},
    {"name": "RandomRain", "action": A.Compose([A.RandomRain(p=1.0)])},
    {"name": "RandomSnow", "action": A.Compose([A.RandomSnow(p=1.0)])},

    {"name": "CoarseDropout", "action": A.Compose([A.CoarseDropout(p=1.0)])},

    {"name": "Sharpen", "action": A.Compose([A.Sharpen(p=1.0)])},
    {"name": "Emboss", "action": A.Compose([A.Emboss(p=1.0)])},
    {"name": "Posterize", "action": A.Compose([A.Posterize(p=1.0)])},
    {"name": "Equalize", "action": A.Compose([A.Equalize(p=1.0)])},
    {"name": "Solarize", "action": A.Compose([A.Solarize(p=1.0)])},
    {"name": "ToGray", "action": A.Compose([A.ToGray(p=1.0)])},
]

logger.info(f"Number of augmentations: {len(augmentations)}")

files = get_input_files()
logger.info(f"Files found: {len(files)}")


def save_image(image_file_name: str, image) -> None:
    """
    Função que salva uma imagem no disco.
    Recebe o nome do arquivo de saída e a imagem (em formato numpy array).
    Não retorna nada, apenas salva a imagem e registra no log.
    Se a imagem não puder ser salva (por exemplo, por falta de permissão),
    registra o erro no log.
    """
    logger.info(f"Saving image: {image_file_name}")
    if not cv2.imwrite(image_file_name, image):
        logger.error(f"Failed to save image: {image_file_name}")



def edit_image(
    input_file_name: str,
    transform,
    output_file_name: str,
) -> None:
    """
    Função que aplica uma transformação da biblioteca Albumentations
    em uma imagem. Recebe o caminho da imagem de entrada, a
    transformação (transform) e o caminho do arquivo de saída.
    Não retorna nada, apenas salva a imagem transformada no disco.
    Se a imagem de entrada não puder ser lida, lança um RuntimeError.
    """
    image = cv2.imread(input_file_name)

    if image is None:
        raise RuntimeError(f"Could not read image: {input_file_name}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = transform(image=image)["image"]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    save_image(
        image_file_name=output_file_name,
        image=image,
    )


def process(image, augmentation) -> None:
    """
    Função que processa uma combinação de imagem e transformação.
    Recebe a imagem (um Path) e a transformação (um dict com "name" e
    "action"). Monta o nome do arquivo de saída, chama edit_image e save_image.
    Se algo der errado (por exemplo, a imagem não puder ser lida), o erro é
    capturado e registrado no log, mas o processamento continua com as outras combinações.
    """
    logger.info(f"{image.name} -> {augmentation['name']}")

    output_file_name = (
        OUTPUT_DIR /
        f"{image.stem}_{augmentation['name']}{image.suffix}"
    )

    try:
        edit_image(
            input_file_name=str(image),
            transform=augmentation["action"],
            output_file_name=str(output_file_name),
        )
    except Exception as exc:
        logger.error(f"Failed on {image.name} -> {augmentation['name']}: {exc}")


for image, augmentation in product(files, augmentations):
    process(image, augmentation)

logger.info("Finished.")
