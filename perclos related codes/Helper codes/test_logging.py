import logging

# Create and configure logger
LOG_FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
logging.basicConfig(filename="E:\\Perclos_k_io\\Logs\\mp_output.log",
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode='w')
logger = logging.getLogger()

# Test the logger
x = 5
logger.info(f"# It works.And x = {x}")

# print(logger.level)

