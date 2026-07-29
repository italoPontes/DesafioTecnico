import cv2

from itertools import product

from augraphy import (
    BadPhotoCopy,
    BindingsAndFasteners,
    BleedThrough,
    BookBinding,
    Brightness,
    BrightnessTexturize,
    ColorPaper,
    ColorShift,
    DelaunayTessellation,
    DepthSimulatedBlur,
    DirtyDrum,
    DirtyRollers,
    DirtyScreen,
    Dithering,
    DotMatrix,
    DoubleExposure,
    Faxify,
    Folding,
    Gamma,
    GlitchEffect,
    Hollow,
    InkBleed,
    InkColorSwap,
    InkMottling,
    InkShifter,
    Jpeg,
    LCDScreenPattern,
    LensFlare,
    Letterpress,
    LightingGradient,
    LinesDegradation,
    LowInkPeriodicLines,
    LowInkRandomLines,
    LowLightNoise,
    Moire,
    Noise,
    NoiseTexturize,
    NoisyLines,
    PageBorder,
    PixelBleed,
    ReflectedLight,
    Scribbles,
    SectionShift,
    ShadowCast,
    Squish,
    Stains,
    SubtleNoise,
    VoronoiTessellation,
    WaterMark,
)

from common import DATABASE_DIR, ensure_output_dir, get_input_files, setup_logger

OUTPUT_DIR = ensure_output_dir(DATABASE_DIR / "augraphy")
logger = setup_logger("augraphy", OUTPUT_DIR)

augmentations = [
    {"name": "BadPhotoCopy", "action": BadPhotoCopy()},
    {"name": "BindingsAndFasteners", "action": BindingsAndFasteners()},
    {"name": "BleedThrough", "action": BleedThrough()},
    {"name": "BookBinding", "action": BookBinding()},
    {"name": "Brightness", "action": Brightness()},
    {"name": "BrightnessTexturize", "action": BrightnessTexturize()},
    {"name": "ColorPaper", "action": ColorPaper()},
    {"name": "ColorShift", "action": ColorShift()},
    {"name": "DelaunayTessellation", "action": DelaunayTessellation()},
    {"name": "DepthSimulatedBlur", "action": DepthSimulatedBlur()},
    {"name": "DirtyDrum", "action": DirtyDrum()},
    {"name": "DirtyRollers", "action": DirtyRollers()},
    {"name": "DirtyScreen", "action": DirtyScreen()},
    {"name": "Dithering", "action": Dithering()},
    {"name": "DotMatrix", "action": DotMatrix()},
    {"name": "DoubleExposure", "action": DoubleExposure()},
    {"name": "Faxify", "action": Faxify()},
    {"name": "Folding", "action": Folding()},
    {"name": "Gamma", "action": Gamma()},
    {"name": "GlitchEffect", "action": GlitchEffect()},
    {"name": "Hollow", "action": Hollow()},
    {"name": "InkBleed", "action": InkBleed()},
    {"name": "InkColorSwap", "action": InkColorSwap()},
    {"name": "InkMottling", "action": InkMottling()},
    {"name": "InkShifter", "action": InkShifter()},
    {"name": "Jpeg", "action": Jpeg()},
    {"name": "LCDScreenPattern", "action": LCDScreenPattern()},
    {"name": "LensFlare", "action": LensFlare()},
    {"name": "Letterpress", "action": Letterpress()},
    {"name": "LightingGradient", "action": LightingGradient()},
    {"name": "LinesDegradation", "action": LinesDegradation()},
    {"name": "LowInkPeriodicLines", "action": LowInkPeriodicLines()},
    {"name": "LowInkRandomLines", "action": LowInkRandomLines()},
    {"name": "LowLightNoise", "action": LowLightNoise()},
    {"name": "Moire", "action": Moire()},
    {"name": "Noise", "action": Noise()},
    {"name": "NoiseTexturize", "action": NoiseTexturize()},
    {"name": "NoisyLines", "action": NoisyLines()},
    {"name": "PageBorder", "action": PageBorder()},
    {"name": "PixelBleed", "action": PixelBleed()},
    {"name": "ReflectedLight", "action": ReflectedLight()},
    {"name": "Scribbles", "action": Scribbles()},
    {"name": "SectionShift", "action": SectionShift()},
    {"name": "ShadowCast", "action": ShadowCast()},
    {"name": "Squish", "action": Squish()},
    {"name": "Stains", "action": Stains()},
    {"name": "SubtleNoise", "action": SubtleNoise()},
    {"name": "VoronoiTessellation", "action": VoronoiTessellation()},
    {"name": "WaterMark", "action": WaterMark()},
]

logger.info(f"Number of modifications: {len(augmentations)}")

files = get_input_files()
logger.info(f"Files found: {len(files)}")


"""
Função para salvar a imagem processada no disco.
Recebe o caminho onde o arquivo vai ser gravado
(ex: "../database/augraphy/foto1_BadPhotoCopy.jpg")
e a imagem já transformada, no formato que o OpenCV
entende (um array BGR).

Não retorna nada, só grava o arquivo e, se o cv2 não conseguir
salvar por algum motivo (pasta sem permissão, formato inválido etc.),
registra o erro no log.
"""
def save_image(image_file_name: str, image) -> None:
    logger.info(f"Saving image to: {image_file_name}")
    if not cv2.imwrite(image_file_name, image):
        logger.error(f"Failed to save image: {image_file_name}")


"""
Função que aplica a transformação de Augraphy na imagem.
Recebe o caminho da imagem de entrada, a transformação (action)
e o caminho do arquivo de saída.
Não retorna nada, só salva o arquivo de saída.
Se a imagem de entrada não puder ser lida, lança um RuntimeError.
"""
def edit_image(
    input_file_name: str,
    action,
    output_file_name: str,
) -> None:
    input_image = cv2.imread(input_file_name)

    if input_image is None:
        raise RuntimeError(f"Could not read image: {input_file_name}")

    result = action(input_image)

    if isinstance(result, dict):
        result = result["output"]

    save_image(
        image_file_name=output_file_name,
        image=result,
    )


"""
Função que processa uma combinação de imagem e transformação.
Recebe a imagem (um Path) e a transformação (um dict com "name" e "action").
Monta o nome do arquivo de saída, chama edit_image e save_image.
Se algo der errado (por exemplo, a imagem não puder ser lida), o erro é
capturado e registrado no log, mas o processamento continua com as outras combinações.  
"""
def process(image, augmentation) -> None:
    logger.info(f"{image.name} -> {augmentation['name']}")

    output_file_name = (
        OUTPUT_DIR / f"{image.stem}_{augmentation['name']}{image.suffix}"
    )

    try:
        edit_image(
            input_file_name=str(image),
            action=augmentation["action"],
            output_file_name=str(output_file_name),
        )
    except Exception as exc:
        logger.error(f"Failed on {image.name} -> {augmentation['name']}: {exc}")


for image, augmentation in product(files, augmentations):
    process(image, augmentation)

logger.info("Finished.")
