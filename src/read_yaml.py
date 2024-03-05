import yaml

def read_yaml(file_path):
    stream = open(file_path, 'r')
    dict = yaml.load(stream, Loader=yaml.FullLoader)
    return dict