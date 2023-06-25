from settings import *
import cv2
import face_recognition
import math
from scipy.spatial import distance as dist
import dlib
from imutils import face_utils
import numpy as np
import os
import sys
import time
from pynput.mouse import Button, Controller

mouse = Controller()

known_face_encodings = []
known_face_names = []

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(EOLHOINICIO,EOLHOFIM) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(DOLHOINICIO,DOLHOFIM) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

counter_dir = 0
counter_esq = 0
counter_boca_aberta = 0
counter_boca_fechada = 0

boca_aberta =False
pressionado = False


class Mouse:
    mouse = Controller()
    def __init__(self):
        print('Classe Mouse')

    def clique_direito(self):
        mouse.press(Button.right)
        mouse.release(Button.right)

    def clique_esquerdo(self):
        mouse.press(Button.left)
        if not boca_aberta:
            mouse.release(Button.left)
        else:
            global pressionado
            pressionado=True

    def solta_botao_esquerdo(self):
        mouse.release(Button.left)

    def duplo_clique_esquerdo(self):
        mouse.click(Button.left, 2)
        mouse.release(Button.left)

    def move_cima(self):
        time.sleep(Settings.PAUSA)
        mouse.move(0,Settings.DISTANCIA*-1)

    def move_baixo(self):
        time.sleep(Settings.PAUSA)
        mouse.move(0,Settings.DISTANCIA)

    def move_direita(self):
        time.sleep(Settings.PAUSA)
        mouse.move(Settings.DISTANCIA,0)

    def move_esquerda(self):
        time.sleep(Settings.PAUSA)
        mouse.move(Settings.DISTANCIA*-1,0)

class Face:

    def __init__(self,__webcan_index=0):
        self.__webcan_index = __webcan_index
        self.__direcao_h = 'centro horizontal'
        self.__direcao_v = 'centro vertical'
        self.__ponta_nariz = 0

    def define_webcam(self, index):
        self.__webcan_index = index

    @property
    def webcan_index(self):
        return self.__webcan_index

    @webcan_index.setter
    def webcan_index(self, webcan_index):
        self.__webcan_index = webcan_index

    def get_webcan_index(self):
        return self.__webcan_index

    def set_webcan_index(self, webcan_index):
        self.__webcan_index = webcan_index

    @property
    def rosto(self):
        global known_face_encodings
        global known_face_names
        return [known_face_encodings,known_face_names]

    @rosto.setter
    def rosto(self, path, name):
        rosto_image = face_recognition.load_image_file(path)
        rosto_face_encoding = face_recognition.face_encodings(rosto_image)[0]
        global known_face_encodings
        global known_face_names
        known_face_encodings.append(rosto_face_encoding)
        known_face_names.append(name)

    def get_rosto(self):
        global known_face_encodings
        global known_face_names
        return [known_face_encodings,known_face_names]

    def set_rosto(self, path, nome):
        rosto_image = face_recognition.load_image_file(path)
        rosto_face_encoding = face_recognition.face_encodings(rosto_image)[0]
        global known_face_encodings
        global known_face_names
        known_face_encodings.append(rosto_face_encoding)
        known_face_names.append(nome)

    def direcao_horizontal(self, max, min):
        if (math.fabs(max- min)) < Settings.INCLINACAO_H:
            self.__direcao_h = "centro horizontal"
        elif max > min:
            Mouse.move_esquerda(self)
            self.__direcao_h = "esquerda"
        else:
            Mouse.move_direita(self)
            self.__direcao_h = "direita"

    @property
    def direcao_h(self):
        return self.__direcao_h

    @direcao_h.setter
    def direcao_h(self, direcao):
        global __direcao_h
        __direcao_h= direcao

    def get_direcao_h(self):
        return self.__direcao_h

    def set_direcao_h(self, direcao):
        global __direcao_h
        __direcao_h = direcao

    @property
    def ponta_nariz(self):
        return self.__ponta_nariz

    @ponta_nariz.setter
    def ponta_nariz(self, ponta_nariz):
        self.__ponta_nariz = ponta_nariz

    def get_ponta_nariz(self):
        return self.__ponta_nariz

    def set_ponta_nariz(self, ponta_nariz):
        self.__ponta_nariz = ponta_nariz

    def direcao_vertical(self,tip):
        if math.fabs(self.get_ponta_nariz() -tip[1]) > Settings.INCLINACAO_V :
            if self.get_ponta_nariz() > tip[1]:
                Mouse.move_cima(self)
                self.__direcao_v = 'cima'
            else:
                Mouse.move_baixo(self)
                self.__direcao_v = 'baixo'
        else:
            self.__direcao_v = 'centro vertical'

    @property
    def direcao_v(self):
        return self.__direcao_v

    @direcao_v.setter
    def direcao_v(self, direcao):
        global __direcao_v
        __direcao_v= direcao

    def get_direcao_v(self):
        return self.__direcao_v

    def set_direcao_v(self, direcao):
        global __direcao_v
        __direcao_v = direcao

    def get_abertura_olho(self,olho):
        A = dist.euclidean(olho[1], olho[5])
        B = dist.euclidean(olho[2], olho[4])
        C = dist.euclidean(olho[0], olho[3])
        return (A + B) / (2.0 * C)

    def piscou_olhos(self,shape):
        global counter_dir
        global counter_esq

        if counter_dir >= Settings.QUANTIDADE_DE_FRAMES and counter_esq >= Settings.QUANTIDADE_DE_FRAMES:
            Mouse.duplo_clique_esquerdo(self)
            counter_dir = 0
            counter_esq = 0

        abertura_d = self.get_abertura_olho(shape[DOLHOINICIO:DOLHOFIM])
        if abertura_d < Settings.ABERTURA_DO_OLHO:
            counter_dir += 1
            if counter_dir > Settings.QUANTIDADE_DE_FRAMES:
                Mouse.clique_direito(self)
                counter_dir = 0
        else:
            if counter_dir >= Settings.QUANTIDADE_DE_FRAMES:
                Mouse.clique_direito(self)
            counter_dir = 0

        abertura_e=self.get_abertura_olho(shape[EOLHOINICIO:EOLHOFIM])
        if abertura_e < Settings.ABERTURA_DO_OLHO:
                counter_esq += 1
                if counter_esq > Settings.QUANTIDADE_DE_FRAMES:
                    Mouse.clique_esquerdo(self)
                    counter_esq = 0
        else:
            if counter_esq >= Settings.QUANTIDADE_DE_FRAMES:
                Mouse.clique_esquerdo(self)
            counter_esq = 0

    def boca_aberta(self,top_lip,bottom_lip):
        global counter_boca_aberta
        global counter_boca_fechada
        global boca_aberta
        distancia = (dist.euclidean(top_lip[8], bottom_lip[10])+dist.euclidean(top_lip[10], bottom_lip[8]))/2
        if distancia >= Settings.BOCA_ABERTA:
            counter_boca_aberta += 1
            if counter_boca_aberta >= Settings.QUANTIDADE_DE_FRAMES:
                boca_aberta =True
                counter_boca_fechada = 0
        else:
            counter_boca_fechada += 1
            if counter_boca_fechada >= Settings.QUANTIDADE_DE_FRAMES:
                counter_boca_aberta = 0
                boca_aberta =False
                global pressionado
                if(pressionado):
                    Mouse.solta_botao_esquerdo(self)
                    pressionado=False

    def get_detector(self,imagem, n):
        return DETECTOR(imagem,n)

    def get_predictor(self, imagem, rect):
        return PREDICTOR(imagem, rect)
