import logging
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from glyphs import ALPHABETS


def save_image_grid(path, np_array, n_cols=None):
    n_imgs, h, w = np_array.shape
    n_cols = n_cols or int(np.ceil(np.sqrt(n_imgs)))
    n_rows = int(np.ceil(n_imgs / n_cols))
    np_array = np.concatenate([np_array, np.zeros((n_cols * n_rows - n_imgs, h, w))])
    rows = [np.concatenate(row, axis=1) for row in np.split(np_array, n_cols)]
    grid = np.concatenate(rows)
    Image.fromarray(grid).convert("RGB").save(path)


def save_message(path, np_array, n_cols=None):
    n_imgs, h, w = np_array.shape
    message = [
        ['F_', 'u', 'n',],
        ['W_', 'i', 't', 'h',],
        ['G_', 'A_', 'N_', 's'],
    ]
    max_len = max([len(word) for word in message])

    message_img = []
    for word in message:
        word_img = []
        for t in word:
            word_img.append(np_array[ALPHABETS.index(t)])
        while len(word_img) < max_len:
            word_img.append(np.zeros((h, w)))
        word_img = np.concatenate(word_img, axis=1)
        message_img.append(word_img)
    message_img = np.concatenate(message_img)
    Image.fromarray(message_img).convert("RGB").save(path)


def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, f'{name}.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
