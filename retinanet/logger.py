import logging


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger('retinanet')
    if logger.hasHandlers():
        return logger

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger
