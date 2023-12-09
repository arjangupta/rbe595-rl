import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

AERIEL_GYM_PATH = os.environ.get("AERIEL_GYM_PATH")

if AERIEL_GYM_PATH is None:
    raise Exception("Please set the AERIEL_GYM_PATH environment variable")
else:
    print("AERIEL_GYM_PATH set as:", AERIEL_GYM_PATH)

def get_aerial_gym_path():
    return AERIEL_GYM_PATH

def get_aerial_gym_envs_base_path():
    return join(AERIEL_GYM_PATH, "aerial_gym/envs/base/")