import logging

level = logging.INFO
logging.basicConfig(filename="log.txt", level=level, format="%(asctime)s - [%(levelname)s] - %(message)s")
