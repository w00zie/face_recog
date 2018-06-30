import pickle
from keras.models import Model
import time
import configparser
import requests
import os


def load_resnet():
    from keras.applications.resnet50 import ResNet50
    realmodel = ResNet50(weights='imagenet')
    print(realmodel.summary())
    realmodel = Model(input=realmodel.layers[0].input, output=realmodel.layers[-2].output)
    print(realmodel.summary())
    return realmodel


def timing(func):
    # @functools.wraps(func)
    def newfunc(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time() - start
        print("function '{}' finished in {} s".format(
            func.__name__, end))
        return res
    return newfunc


def pickle_stuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()


def load_stuff(filename):
    saved_stuff = open(filename, "rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


def download_files(url, name):
    r = requests.get(url, allow_redirects = True)
    open(name, 'wb').write(r.content)


def check_file(path, message, url):
    local_path = path
    while not os.path.isfile(local_path):

        local_path = input(message)

        if local_path == '':
            local_path = path
            download_files(url, local_path)

    return local_path


class Configuration:

    def __init__(self,
                 thresh = 0.35,
                 conf = 8.,
                 performance = 0,
                 haar = 'haarcascade_frontalface_default.xml',
                 vgg = 'vgg-face.mat',
                 video = ""):

        self.name = 'DEFAULT'
        self.config_path = 'config.ini'
        self.threshold = thresh
        self.confidence = conf
        self.performance = performance
        self.haar_path = haar
        self.vgg_path = vgg
        self.video_path = video
        self.config = configparser.ConfigParser()

    def check_requirements(self):
        haar_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/' \
                   'haarcascades/haarcascade_frontalface_default.xml'
        vgg_url = 'http://www.vlfeat.org/matconvnet/models/vgg-face.mat'

        message = 'Please insert path to face detector, leave empty to download default (haar):\n'

        self.haar_path = check_file(self.haar_path, message, haar_url)

        message = 'Please insert path to pre-trained model, ' \
                  'leave empty to download default (VGG16):\n'
        if self.vgg_path = '':
            self.vgg_path = 'vgg_face.mat'
        self.vgg_path = check_file(self.vgg_path, message, vgg_url)

    def set_variables(self):
        self.threshold = float(self.config[self.name]['threshold'])
        self.confidence = float(self.config[self.name]['confidence'])
        self.haar_path = self.config[self.name]['haar_path']
        self.vgg_path = self.config[self.name]['vgg_path']

        self.check_requirements()

        if self.haar_path != self.config[self.name]['haar_path']:
            self.config[self.name]['haar_path'] = self.haar_path
            with open(self.config_path, 'w') as file:
                self.config.write(file)

        if self.vgg_path != self.config[self.name]['vgg_path']:
            self.config[self.name]['vgg_path'] = self.vgg_path
            with open(self.config_path, 'w') as file:
                self.config.write(file)

        self.video_path = self.config[self.name]['video_path']
        self.performance = int(self.config[self.name]['performance'])

    def write_config(self):
        print("Creating a new configuration file...")
        self.check_requirements()

        self.config_path = input('Insert your configuration file name:\n')
        self.config_path = '{}.ini'.format(self.config_path)

        self.config[self.name] = {'threshold': self.threshold,
                                  'confidence': self.confidence,
                                  'haar_path': self.haar_path,
                                  'vgg_path': self.vgg_path,
                                  'video_path': self.video_path,
                                  'performance': self.performance}

        with open(self.config_path, 'a') as file:
            self.config.write(file)

    def read_config(self):
        print("Searching for a configuration file in current folder...")

        conf_files = [f for f in os.listdir('.') if f.endswith('.ini')]

        if len(conf_files) != 1:

            if len(conf_files) > 1:
                for i in range(len(conf_files)):
                    os.remove(conf_files[i])
                print("Too many configuration files, they'll be deleted!")
            else:
                print("No configuration file found.")

            self.write_config()
            self.config.read(self.config_path)
        else:
            self.config_path = conf_files[0]
            self.config.read(self.config_path)
            profiles = [profile for profile in self.config]

            index = -1
            while index not in range(len(profiles)):
                for i in range(len(profiles)):
                    print("{} - {}".format(i, profiles[i]))
                try:
                    index = int(input("Please select your profile:\n"))
                except ValueError:
                    print("Please insert a valid value!\n")

            self.name = profiles[index]

        self.set_variables()
