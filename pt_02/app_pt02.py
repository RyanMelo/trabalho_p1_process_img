import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

imagem = cv2.imread("pt_02/imagem.png")
imagem_RGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# Preencher todos os buracos dos objetos pretos
def preencherBuracos():
    kernel = np.ones((5, 5), np.uint8)
    ImagemBuracosPreenchidos = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, kernel)

    cv2.imshow("Imagem Original", imagem)
    cv2.imshow("Imagem Preenchida", ImagemBuracosPreenchidos)
    cv2.waitKey(0)

# Eliminiar todos e somente os objetos pretos
def eliminarObjetosPretos():
    imagem_invertida = cv2.bitwise_not(imagem)

    kernel = np.ones((5, 5), np.uint8)

    imagemFechada = cv2.morphologyEx(imagem_invertida, cv2.MORPH_CLOSE, kernel)

    imagem_preenchida = cv2.bitwise_not(imagemFechada)

    mascaraObjetosPretos = (imagem_preenchida == [0, 0, 0]).all(axis=2)

    imagemSemObjetos_pretos = imagem.copy()

    imagemSemObjetos_pretos[mascaraObjetosPretos] = [255, 255, 255]

    cv2.imshow("Imagem Original", imagem)
    cv2.imshow("Imagem sem objetos pretos", imagemSemObjetos_pretos)
    cv2.waitKey(0)

# Preencher os buracos dos objetos de cor azul, amarelo e verde
def preecherBuracosObjetos():
    print("Ainda nao fiz")

# Realçar as bordas da imagem
def realcarBordas():
    imagemBordasRealcadas = cv2.Canny(imagem, threshold1=30, threshold2=70)

    cv2.imshow('Imagem Original', imagem)
    cv2.imshow('Imagem Com Bordas Realcadas', imagemBordasRealcadas)
    cv2.waitKey(0)
