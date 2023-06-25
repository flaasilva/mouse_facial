import imutils

from utils import *
import cv2
from imutils import face_utils

def main():

    face_util = Face()
    face_util.define_webcam(2)

    face_util.set_rosto("flavio.jpg","Flavio")
    video_capture = cv2.VideoCapture(face_util.get_webcan_index())

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding, face_landmarks in zip(face_encodings, face_landmarks_list):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                nose_tip = face_landmarks['nose_tip']
                maxLandmark = max(nose_tip)
                minLandmark = min(nose_tip)
                face_util.direcao_horizontal(maxLandmark[1], minLandmark[1])

                maxTip = nose_tip[3]

                if face_util.get_ponta_nariz() == 0:
                    face_util.set_ponta_nariz(maxTip[1])
                face_util.direcao_vertical(maxTip)

                face_util.boca_aberta(face_landmarks['top_lip'], face_landmarks['bottom_lip'])

            face_names.append("{} {} {}".format(name, face_util.get_direcao_h(), face_util.get_direcao_v()))

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                if 'Unknown' not in name.split(' '):
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = face_util.get_detector(gray, 0)
                for rect in rects:
                    shape = face_util.get_predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    face_util.piscou_olhos(shape)

        frame = cv2.flip(frame, 1)
        cv2.namedWindow("video", cv2.WINDOW_NORMAL)
        cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print('iniciou')
    main()
