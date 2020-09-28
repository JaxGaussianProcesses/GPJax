import yaml


class Colours:
    def __init__(self, path):
        with open(path) as f:
            # use safe_load instead load
            dataMap = yaml.safe_load(f)
        cols = dataMap['colours']

        colours = {}
        for c in cols:
            colours[list(c.keys())[0]] = list(c.values())[0]

        for k, v in colours.items():
            setattr(self, k, v)


if __name__ == '__main__':
    c = Colours('config_files/defaults.yaml')
    print(c.primary)