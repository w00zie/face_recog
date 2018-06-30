import cv2
import arch
from scipy.spatial.distance import cosine as dcos
import utils
import os.path


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
            self.known_people = utils.load_stuff("known.pickle")
        except IOError:
            print("No known people")
            self.known_people = []

        self.seen_people = []
        self.images = []

    @utils.timing
    def pred_img(self, crop_img):
        crop_img = cv2.resize(crop_img, (self.face_size, self.face_size))
        out = arch.my_pred(self.__realmodel, crop_img, transform = True)
        return out

    def check_index(self, fvec, pred_list):
        index = 0
        cosdismin = 1
        while cosdismin >= self.threshold and index < len(pred_list):
            try:
                cosdis = dcos(fvec, pred_list[index]['face'])
            except IndexError:
                cosdis = dcos(fvec, pred_list[index])
            if cosdis < cosdismin:
                cosdismin = cosdis
            index += 1
        return index, cosdismin

    def prediction(self, crop_img):
        # cv2.imshow("cropped image", crop_img)
        fvec = self.pred_img(crop_img)[0, :]
        known = False
        know_index = 0
        seen_index = 0
        if len(self.seen_people) == 0 and len(self.known_people) == 0:
            self.seen_people.append(fvec)
        else:
            know_index, cosdismin_y = self.check_index(fvec, self.known_people)

            if cosdismin_y >= self.threshold:
                seen_index, cosdismin_k = self.check_index(fvec, self.seen_people)

                if cosdismin_k >= self.threshold:
                    self.seen_people.append(fvec)
                    print("face {} is new".format(seen_index))
                    seen_index = len(self.seen_people)

            else:
                known = True

            seen_index -= 1
            know_index -= 1
        return seen_index, know_index, known

    def get_faces(self):
        # Capture frame-by-frame
        ret, frame = self.__video_capture.read()
        # frame_num = __video_capture.get(1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, cose, conf = self.faceCascade.detectMultiScale3(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE,
            outputRejectLevels = True
        )

        return frame, faces, conf

    def save_faces(self):
        for i in range(len(self.seen_people)):
            cv2.imshow('face', self.images[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            name = input('Choose a name for this person (leave empty to discard face):\n')
            if len(name):
                self.known_people.append({'name': name, 'face': self.seen_people[i]})

    def check_faces(self, known, seen_id, know_id, old_len_faces, checked_faces):
        frame, faces, conf = self.get_faces()

        text = "Seen {} different people".format(len(self.seen_people) + len(self.known_people))
        cv2.putText(frame, text, (60, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        if len(faces) > old_len_faces:
            self.predict = True
        if len(faces) < checked_faces:
            checked_faces = 0
        for i, (x, y, w, h) in enumerate(faces):
            old_seen = len(self.seen_people)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            crop_img = frame[y:y + h, x:x + w]

            if self.performance:
                if conf[i] >= self.confidence / 2:
                    seen_id, know_id, known = self.prediction(crop_img)
                    checked_faces += 1
                if known:
                    cv2.putText(frame,
                                "{}".format(self.known_people[know_id]['name']),
                                (x, y - 3),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (125, 0, 0),
                                2)
                else:
                    try:
                        if seen_id >= old_seen:
                            self.images.append(crop_img)
                    except TypeError:
                        pass
                    cv2.putText(frame, "New person {}".format(seen_id), (x, y - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 125), 2)
            else:
                if self.predict and conf[i] >= self.confidence:
                    seen_id, know_id, known = self.prediction(crop_img)
                    checked_faces += 1
                if known:
                    cv2.putText(frame,
                                "Last recognized: {}".format(self.known_people[know_id]['name']),
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 0),
                                2)
                else:
                    try:
                        if seen_id >= old_seen:
                            self.images.append(crop_img)
                    except TypeError:
                        pass
                    cv2.putText(frame, "Last seen: new person {}".format(seen_id), (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 125), 2)

                if checked_faces == len(faces):
                    self.predict = False

        return frame, faces, seen_id, know_id, known, checked_faces

    def loop_frames(self):
        old_len_faces = 0
        checked_faces = 0
        seen_id = '?'
        know_id = '?'
        known = False

        while True:

            frame, faces, seen_id, know_id, known, checked_faces = self.check_faces(
                known, seen_id, know_id, old_len_faces, checked_faces)

            # Display the resulting frame
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            old_len_faces = len(faces)

        # When everything is done, release the capture
        self.__video_capture.release()
        cv2.destroyAllWindows()
        if len(self.seen_people):
            self.save_faces()
        utils.pickle_stuff("known.pickle", self.known_people)
