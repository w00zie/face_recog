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

def colors(n):
    import random
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(n)]
    return color

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
        while self.threshold <= 0 or self.threshold >= 1:
            try:
                self.threshold = float(input("Please choose a threshold (0, 1):\n"))
            except ValueError:
                print("Please insert a valid value!\n")
        while self.confidence < 0 or self.confidence > 10:
            try:
                self.confidence = float(input("Please choose a confidence [0, 10]:\n"))
            except ValueError:
                print("Please insert a valid value!\n")
        while self.performance != 0 and self.performance != 1:
            try:
                self.performance = int(input("Do you want to increase performance (0 - no, 1 - yes):\n"))
            except ValueError:
                print("Please insert a valid value!\n")

        haar_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/' \
                   'haarcascades/haarcascade_frontalface_default.xml'
        vgg_url = 'http://www.vlfeat.org/matconvnet/models/vgg-face.mat'

        message = 'Please insert path to face detector, leave empty to download default (haar):\n'

        self.haar_path = check_file(self.haar_path, message, haar_url)

        message = 'Please insert path to pre-trained model, ' \
                  'leave empty to download default (VGG16):\n'
        self.vgg_path = check_file(self.vgg_path, message, vgg_url)

    def set_variables(self):
        try:
            self.threshold = float(self.config[self.name]['threshold'])
        except KeyError:
            self.threshold = 0.35
            self.config[self.name]['threshold'] = self.threshold
        try:
            self.confidence = float(self.config[self.name]['confidence'])
        except KeyError:
            self.confidence = 8.
            self.config[self.name]['confidence'] = self.confidence
        try:
            self.haar_path = self.config[self.name]['haar_path']
        except KeyError:
            self.haar_path = "haarcascade_frontalface_default.xml"
            self.config[self.name]['haar_path'] = self.haar_path
        try:
            self.vgg_path = self.config[self.name]['vgg_path']
        except KeyError:
            self.vgg_path = "vgg-face.mat"
            self.config[self.name]['vgg_path'] = self.vgg_path
        try:
            self.video_path = self.config[self.name]['video_path']
        except KeyError:
            self.video_path = ""
            self.config[self.name]['video_path'] = ""
        try:
            self.performance = int(self.config[self.name]['performance'])
        except KeyError:
            self.performance = 0
            self.config[self.name]['performance'] = 0

        self.check_requirements()

        self.config[self.name] = {'threshold': self.threshold,
                                  'confidence': self.confidence,
                                  'haar_path': self.haar_path,
                                  'vgg_path': self.vgg_path,
                                  'video_path': self.video_path,
                                  'performance': self.performance}

        with open(self.config_path, 'w') as file:
            self.config.write(file)

    def write_config(self, custom = False):
        print("Creating a new configuration...")

        self.check_requirements()

        if not custom:
            self.config_path = input('Insert your configuration file name:\n')
            self.config_path = '{}.ini'.format(self.config_path)

        name = ""

        while not len(name):
            name = input('Choose a profile name:\n')

        self.name = name

        self.config[self.name] = {'threshold': self.threshold,
                                  'confidence': self.confidence,
                                  'haar_path': self.haar_path,
                                  'vgg_path': self.vgg_path,
                                  'video_path': self.video_path,
                                  'performance': self.performance}

        with open(self.config_path, 'w') as file:
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
        else:
            self.config_path = conf_files[0]
            self.config.read(self.config_path)
            profiles = [profile for profile in self.config]

            if not len(profiles):
                self.write_config(True)
            else:
                index = -1
                while index not in range(len(profiles) + 1):
                    for i in range(len(profiles)):
                        print("{} - {}".format(i, profiles[i]))
                    print("{} - CUSTOM".format(len(profiles)))
                    try:
                        index = int(input("Please select your profile:\n"))
                    except ValueError:
                        print("Please insert a valid value!\n")

                if index == len(profiles):
                    self.write_config(True)
                else:
                    self.name = profiles[index]
                    self.set_variables()
