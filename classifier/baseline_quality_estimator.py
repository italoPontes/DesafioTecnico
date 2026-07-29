import cv2
import numpy as np

def estimate_sharpness(image_path: str) -> float:
    """
    Estima a nitidez de uma imagem.

    Parameters
    ----------
    image_path : str
        Caminho da imagem.

    Returns
    -------
    float
        Score entre 0 e 100.
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Não foi possível abrir a imagem: {image_path}")

    # ---------------------------------------------------
    # 1) Variância do Laplaciano
    # ---------------------------------------------------

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    lap_var = laplacian.var()

    # ---------------------------------------------------
    # 2) Intensidade média do gradiente (Sobel)
    # ---------------------------------------------------

    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    gradient = np.sqrt(gx**2 + gy**2)
    grad_mean = gradient.mean()

    # ---------------------------------------------------
    # Combinação das métricas
    # ---------------------------------------------------

    # Valores típicos
    #
    # Laplacian:
    #   Blur intenso ............. 10~30
    #   Médio .................... 50~150
    #   Boa nitidez .............. 200~500
    #   Muito nítida ............. >700
    #
    # Sobel:
    #   Blur ..................... <15
    #   Normal ................... 20~40
    #   Muito nítida ............. >50

    lap_score = np.clip(lap_var / 600.0, 0, 1)
    grad_score = np.clip(grad_mean / 50.0, 0, 1)

    score = 100 * (0.7 * lap_score + 0.3 * grad_score)

    return float(round(score, 2))



