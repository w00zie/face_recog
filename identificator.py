import cv2
import arch
import utils
import os.path
from cluster import Cluster


class Identificator:

    def __init__(self, confidence, threshold, haar_path, vgg_path, performance = False,
                 video_path = None):

        self.face_size = 224  # Size of the crop for the face
        self.predict = True  # True if faces need to be identified, False otherwise
        self.resnet = False  # Whether to use an alternative network
        self.threshold = threshold
        self.confidence = confidence
        self.performance = performance

        if self.resnet:
            self.__realmodel = utils.load_resnet()
        else:
            self.__realmodel = arch.get_model(vgg_path)

        self.faceCascade = cv2.CascadeClassifier(haar_path)

        if video_path is None or not os.path.isfile(video_path):
            self.__video_capture = cv2.VideoCapture(0)
        else:
            self.__video_capture = cv2.VideoCapture(video_path)

        try:
            self.cluster = utils.load_stuff("known.pickle")
        except IOError:
            print("No known people")
            self.cluster = Cluster()

        self.images = []

    @utils.timing
    def pred_img(self, crop_img):
        crop_img = cv2.resize(crop_img, (self.face_size, self.face_size))
        out = arch.my_pred(self.__realmodel, crop_img, transform = True)
        return out

    def get_faces(self):
        # Capture frame-by-frame
        ret, frame = self.__video_capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces, _, conf = self.faceCascade.detectMultiScale3(
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE,
                outputRejectLevels = True
            )
            return frame, faces, conf
        else:
            print("Video ended")
            return None

    def save_faces(self):
        i = 0
        while i < self.cluster.node_idx:
            print("i = {}, node_idx = {}".format(i, self.cluster.node_idx))
            if isinstance(self.cluster.G.node[i]['name'], int):
                cv2.imshow('face', self.images[self.cluster.G.node[i]['name']])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                name = input('Choose a name for this person (leave empty to discard face):\n')
                if len(name):
                    self.cluster.add_name(name)
                    i += 1
                else:
                    self.images.pop(self.cluster.G.node[i]['name'])
                    self.cluster.clear_class(self.cluster.G.node[i]['name'])
            else:
                i += 1

    def check_faces(self, old_len_faces, checked_faces):
        try:
            frame, faces, conf = self.get_faces()
        except TypeError:
            self.close_video()
            exit()
        text = "Seen {} different people".format(len(self.cluster.people_idx))
        cv2.putText(frame, text, (60, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        if len(faces) > old_len_faces:
            self.predict = True
        if len(faces) < checked_faces:
            checked_faces = 0
        nclust = len(self.cluster.people_idx)
        for i, (x, y, w, h) in enumerate(faces):
            # old_seen = len(self.seen_people)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            crop_img = frame[y:y + h, x:x + w]

            if self.performance:
                if conf[i] >= self.confidence / 2:
                    self.cluster.update_graph(desc = self.pred_img(crop_img)[0, :])
                    checked_faces += 1
                try:
                    if isinstance(self.cluster.nodes[-1][1]['name'], str):
                        cv2.putText(frame,
                                    "{}".format(self.cluster.nodes[-1][1]['name']),
                                    (x, y - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (125, 0, 0),
                                    2)
                    else:
                        try:
                            if self.cluster.people_idx[self.cluster.nodes[-1][1]['name']] == 1:
                                self.images.append(crop_img)
                        except TypeError:
                            pass
                        cv2.putText(frame,
                                    "Person {}".format(self.cluster.nodes[-1][1]['name']),
                                    (x, y - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 125), 2)
                except IndexError:
                    pass
            else:
                if self.predict and conf[i] >= self.confidence:
                    self.cluster.update_graph(desc = self.pred_img(crop_img)[0, :])
                    checked_faces += 1
                try:
                    if isinstance(self.cluster.nodes[-1][1]['name'], str):
                        cv2.putText(frame,
                                    "Last recognized: {}".format(self.cluster.nodes[-1][1]['name']),
                                    (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 0),
                                    2)
                    else:
                        if len(self.cluster.people_idx) > nclust:
                            self.images.append(crop_img)
                        cv2.putText(frame,
                                    "Last seen: Person {}".format(self.cluster.nodes[-1][1]['name']),
                                    (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 125), 2)
                except IndexError:
                    pass

                if checked_faces == len(faces):
                    self.predict = False

        return frame, faces, checked_faces

    def loop_frames(self):
        old_len_faces = 0
        checked_faces = 0

        while True:

            frame, faces, checked_faces = self.check_faces(old_len_faces, checked_faces)

            # Display the resulting frame
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            old_len_faces = len(faces)
        self.close_video()


    def close_video(self):

        # When everything is done, release the capture
        self.__video_capture.release()
        cv2.destroyAllWindows()
        print("Graph = {}".format(self.cluster.G.nodes.data()))
        if self.cluster.node_idx > 0:
            self.save_faces()
        if self.cluster.node_idx > 0:
            self.cluster.plot_graph()
            utils.pickle_stuff("known.pickle", self.cluster)
