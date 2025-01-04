import yaml

class Configurations:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)