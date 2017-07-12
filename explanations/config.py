import os

import yaml


class Config(object):
    """Represents a configuration file for producing explanations.

    """

    def __init__(self, config_dict=None):
        self.short_name = ''
        self.description = ''
        self.output_dir = './s/'

        if config_dict:
            self.__dict__.update(config_dict)

    @property
    def log_path(self):
        return os.path.join(self.output_dir, 'main.log')

    @classmethod
    def from_file(cls, path):
        """Creates an instance of a Config object from a given file path.

        Args:
            path: Path to a YAML configuration file.

        Returns:
            An instance of a Config object.
        """

        config_dict = yaml.load(open(path))
        return cls(config_dict=config_dict)

    def to_yaml(self, path):
        """
        Saves this configuration instance to a YAML file.
        """
        with open(path, 'w') as f:
            yaml.dump(self.__dict__(), f)
