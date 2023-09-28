import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

imagem = cv2.imread("pt_01/lena.png")
imagem_RGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# Redimecionamento
def imagem_redimencionada():
    imagem_redimencionada = cv2.resize(imagem, (256, 256))
    cv2.imshow("Imagem Original", imagem)
    cv2.imshow("Imagem Redimencionada", imagem_redimencionada)
    cv2.waitKey(0)

# Conversão para HSV
def conversão_HSV():
    imagem_hsv = cv2.cvtColor(imagem_RGB, cv2.COLOR_BGR2HSV)
    cv2.imshow("Imagem Original", imagem_RGB)
    cv2.imshow("Imagem HSV", imagem_hsv)
    cv2.waitKey(0)

# Histograma Colorida
def histograma_colorido():
    cores = ('b', 'g', 'r')
    for i, color in enumerate(cores):
        histogramaColorido = cv2.calcHist([imagem], [i], None, [256], [0, 256])
        plt.plot(histogramaColorido, color=color)
        plt.xlim([0, 256])

    plt.title("Histograma Imagem Colorida")
    plt.show()

# Histograma em escala de cinza
def histogramaEscalaCinza():
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    histograma = cv2.calcHist([imagemCinza], [0], None, [256], [0, 256])

    plt.figure()
    plt.title('Histograma')
    plt.xlabel('Valor de pixel')
    plt.ylabel('Número de pixels')
    plt.plot(histograma)
    plt.xlim([0, 256])
    plt.show()

# Imagem Equalizada
def imagem_equalizada():
    canais = cv2.split(imagem)
    canais_equalizados = []

    for canal in canais:
        canal_equalizado = cv2.equalizeHist(canal)
        canais_equalizados.append(canal_equalizado)

    imagem_equalizada = cv2.merge(canais_equalizados)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Imagem Original')
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title('Imagem Equalizada')
    plt.imshow(cv2.cvtColor(imagem_equalizada, cv2.COLOR_BGR2RGB))

    plt.show()

# Canais Separados
def canais_separados():
    canais = cv2.split(imagem)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title('Canal Vermelho')
    plt.imshow(canais[2], cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Canal Verde')
    plt.imshow(canais[1], cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Canal Azul')
    plt.imshow(canais[0], cmap='gray')

    plt.show()
