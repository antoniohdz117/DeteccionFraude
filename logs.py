import logging
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent / "LOGS"
LOG_FILE = LOG_DIR / "proyecto_fraude.log"


def configurar_logger(nombre_logger: str, reiniciar: bool = False):
    """
    Configura un logger para el proyecto.

    reiniciar=True  -> borra el log anterior y crea uno nuevo.
    reiniciar=False -> agrega información al log existente.
    """

    LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger(nombre_logger)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Evita duplicar logs si el archivo se ejecuta varias veces
    if logger.handlers:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    modo_archivo = "w" if reiniciar else "a"

    formato = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        mode=modo_archivo,
        encoding="utf-8"
    )
    file_handler.setFormatter(formato)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formato)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger