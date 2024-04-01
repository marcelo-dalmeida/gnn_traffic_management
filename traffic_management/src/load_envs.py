from os.path import dirname, join

import dotenv


def load():
    dotenv.load_dotenv(dotenv_path=join(dirname(__file__), '../.env'))
    dotenv.load_dotenv(dotenv_path=join(dirname(__file__),
                       '../.env.developer'), override=True)
