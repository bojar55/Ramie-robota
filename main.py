import numpy as np
import math
import random
from matplotlib import pyplot as plt
from PIL import Image
import cv2

ilosc_obrazkow = 1000  # przykłady uczące
kat = np.zeros(ilosc_obrazkow * 2)
kat = kat.reshape(ilosc_obrazkow, 2)
wsp = np.zeros(ilosc_obrazkow * 2)
wsp = wsp.reshape(ilosc_obrazkow, 2)
nkat = np.zeros(ilosc_obrazkow * 2)  # znormalizowany kat
nkat = nkat.reshape(ilosc_obrazkow, 2)
nwsp = np.zeros(ilosc_obrazkow * 3)  # znormalizowana wspolrzedna
nwsp = nwsp.reshape(ilosc_obrazkow, 3)
for i in range(ilosc_obrazkow):
    kat[i][0] = random.randint(1, 179)  # alfa
    kat[i][1] = random.randint(1, 179)  # beta

    # katy[i][0] = 30  # alfa
    # katy[i][1] = 50  # beta

    gamma = (180 - kat[i][1]) / 2  # poniewaz trokjkat rownoramienny (gamma, gamma, beta)
    c = 100 * math.sin(math.radians(kat[i][1])) / math.sin(
        math.radians(gamma))  # przeciwprostokątna z twierdzenia sin'usów
    znak_x = 1
    znak_y = 1
    if kat[i][0] <= 90 and gamma >= kat[i][0]:
        znak_x = -1 * znak_x
    if kat[i][0] >= 90 and (kat[i][0] - gamma >= 90):
        znak_y = -1 * znak_y
    wsp[i][0] = znak_x * abs(math.sin(math.radians(abs(kat[i][0] - gamma)))) * c
    wsp[i][1] = znak_y * abs(math.cos(math.radians(abs(kat[i][0] - gamma)))) * c
    nkat[i][0] = kat[i][0] / 225 + 0.1
    nkat[i][1] = kat[i][1] / 225 + 0.1
    nwsp[i][0] = (wsp[i][0] - 50) / 180
    nwsp[i][1] = wsp[i][1] / 240

pnw = 5  # perceptron na warstwe
w1 = np.random.uniform(low=-1, high=1, size=3 * pnw)  # wagi w1 (3= x+y+bias)
w1 = w1.reshape(3, pnw)  # 3 wagi na kazdy perceptron
w2 = np.random.uniform(low=-1, high=1, size=(pnw + 1) * pnw)  # wagi w2 (pnw+bias)
w2 = w2.reshape(pnw + 1, pnw)
wk = np.random.uniform(low=-1, high=1, size=(pnw + 1) * 2)  # wagi w3 (pnw+bias i 2=alfa+beta)
wk = wk.reshape(pnw + 1, 2)
wynikwarstwy = np.zeros(2 * pnw)
wynikwarstwy = wynikwarstwy.reshape(pnw, 2)
koniec = np.zeros(2)
koniec = koniec.reshape(2, 1)
suma = np.zeros(3 * pnw)
suma = suma.reshape(3, pnw)
suma2 = np.zeros(pnw + 1)
suma2 = suma2.reshape(pnw + 1, 1)
suma3 = np.zeros(pnw + 1)
suma3 = suma3.reshape(pnw + 1, 1)
deltak = np.zeros(2)
delta2 = np.zeros(pnw + 1)
delta1 = np.zeros(pnw + 1)
kat_co_100k = np.zeros(ilosc_obrazkow * 2)
kat_co_100k = kat_co_100k.reshape(ilosc_obrazkow, 2)
wsp_co_100k = np.zeros(ilosc_obrazkow * 2)
wsp_co_100k = wsp_co_100k.reshape(ilosc_obrazkow, 2)
itera = 0
roznice_wsp = np.zeros(ilosc_obrazkow)
roznice_kat = np.zeros(ilosc_obrazkow)
suma_roznic_wsp = 0
suma_roznic_kat = 0
for N in range(1000000):
    przyklad = np.random.randint(low=0, high=ilosc_obrazkow - 1)
    # obliczanie wyników
    for i in range(pnw):  # w1
        suma[0][i] = nwsp[przyklad][0] * w1[0][i] + nwsp[przyklad][1] * w1[1][i] + 1 * w1[2][i]
        wynikwarstwy[i][0] = 1 / (1 + math.exp((-1) * suma[0][i]))

    for i in range(pnw):  # w2
        suma[1][i] = w2[pnw][i] * 1
        for j in range(pnw):
            suma[1][i] += wynikwarstwy[j][0] * w2[j][i]
        wynikwarstwy[i][1] = 1 / (1 + math.exp((-1) * suma[1][i]))

    for i in range(2):  # wk
        suma[2][i] = 1 * wk[pnw][i]
        for j in range(pnw):
            suma[2][i] += wynikwarstwy[j][1] * wk[j][i]
        koniec[i] = 1 / (1 + math.exp((-1) * suma[2][i]))

    # liczymy delty od tyłu
    for i in range(2):  # wk
        deltak[i] = (koniec[i] - nkat[przyklad][i]) * koniec[i] * (1 - koniec[i])

    for i in range(pnw):  # w2
        suma2[i] = wk[i][0] * deltak[0] + wk[i][1] * deltak[1]
        delta2[i] = suma2[i] * wynikwarstwy[i][1] * (1 - wynikwarstwy[i][1])

    # suma2[pnw] = wk[pnw][0] * deltak[0] + wk[pnw][1] * deltak[1]
    # delta2[pnw] = suma2[pnw] * 1 * (1 - 1)

    # for i in range(pnw): #w1
    #   suma3[i] = w2[i][0] * delta2[0] + w2[i][1] * delta2[1] #dodać pętle po j=elementow delta2 (iloczyn skalarny w2[i] * delta2
    #   delta1[i] = suma3[i] * wynikwarstwy[i][0] * (1 - wynikwarstwy[i][0])
    for i in range(pnw):  # w1
        suma3[i] = 0
        for j in range(pnw):
            suma3[i] += w2[i][j] * delta2[j]
        delta1[i] = suma3[i] * wynikwarstwy[i][0] * (1 - wynikwarstwy[i][0])
    # poprawienie wag od tylu
    for i in range(2):  # wk
        for j in range(pnw + 1):
            if j < pnw:
                wk[j][i] += (-0.01) * deltak[i] * wynikwarstwy[j][1]
            else:
                wk[j][i] += (-0.01) * deltak[i] * 1

    for i in range(pnw):  # w2
        for j in range(pnw + 1):
            if j < pnw:
                w2[j][i] += (-0.01) * delta2[i] * wynikwarstwy[j][0]
            else:
                w2[j][i] += (-0.01) * delta2[i] * 1

    for i in range(3):  # w1
        for j in range(pnw):
            if i == 2:
                w1[i][j] += (-0.01) * delta1[j] * 1
            else:
                w1[i][j] += (-0.01) * delta1[j] * nwsp[przyklad][i]

    if N%100000 == 0 or N == 999999:  # % 100000 == 0 and N != 0: # co 100 tys sprawdzaj
        # print("wagim ")
        # print(w1)
        # print(w2)
        # print(wk)
        suma_roznic_kat = 0
        itera = itera + 1
        for k in range(ilosc_obrazkow):
            for i in range(pnw):
                suma[0][i] = nwsp[k][0] * w1[0][i] + nwsp[k][1] * w1[1][i] + 1 * w1[2][i]
                wynikwarstwy[i][0] = 1 / (1 + math.exp((-1) * suma[0][i]))

            for i in range(pnw):
                suma[1][i] = w2[pnw][i] * 1
                for j in range(pnw):
                    suma[1][i] += wynikwarstwy[j][0] * w2[j][i]
                wynikwarstwy[i][1] = 1 / (1 + math.exp((-1) * suma[1][i]))

            for i in range(2):
                suma[2][i] = 1 * wk[pnw][i]
                for j in range(pnw):
                    suma[2][i] = wynikwarstwy[j][1] * wk[j][i]
                koniec[i] = 1 / (1 + math.exp((-1) * suma[2][i]))
                kat_co_100k[k][i] = (koniec[i] - 0.1) * 225

            znak_x = 1
            znak_y = 1
            gamma = (180 - kat_co_100k[k][1]) / 2
            c = 100 * math.sin(math.radians(kat_co_100k[k][1])) / math.sin(math.radians(gamma))
            if kat_co_100k[k][0] <= 90 and gamma >= kat_co_100k[k][0]:
                znak_x *= -1
            if kat_co_100k[k][0] >= 90 and (kat_co_100k[k][0] - gamma >= 90):
                znak_y *= -1

            wsp_co_100k[k][0] = znak_x * abs(math.sin(math.radians(abs(kat_co_100k[k][0] - gamma)))) * c
            wsp_co_100k[k][1] = znak_y * abs(math.cos(math.radians(abs(kat_co_100k[k][0] - gamma)))) * c

        sr_x = 0
        sr_y = 0
        sr_alfa = 0
        sr_beta = 0

        # for i in range(ilosc_obrazkow):
        #   sr_x += wsp_co_100k[i][0] - wsp[i][0]
        #   sr_y += wsp_co_100k[i][1] - wsp[i][1]
        #
        # for i in range(ilosc_obrazkow):
        #   sr_alfa += kat_co_100k[i][0] - kat[i][0]
        #   sr_beta += kat_co_100k[i][1] - kat[i][1]

        for i in range(ilosc_obrazkow):
            roznice_wsp[i] = math.sqrt(((wsp_co_100k[i][0] - wsp[i][0]) * (wsp_co_100k[i][0] - wsp[i][0])) +
                                       ((wsp_co_100k[i][1] - wsp[i][1]) * (wsp_co_100k[i][1] - wsp[i][1])))
            suma_roznic_wsp += roznice_kat[i]

        for i in range(ilosc_obrazkow):
            roznice_kat[i] = math.sqrt(((kat_co_100k[i][0] - kat[i][0]) * (kat_co_100k[i][0] - kat[i][0])) +
                                       ((kat_co_100k[i][1] - kat[i][1]) * (kat_co_100k[i][1] - kat[i][1])))
            suma_roznic_kat += roznice_kat[i]

        print("wynik: ", itera)
        # print("sredni blad x: ", sr_x / ilosc_obrazkow)
        # print("sredni blad y: ", sr_y / ilosc_obrazkow)
        # print("sredni blad alfa: ", sr_alfa / ilosc_obrazkow)
        # print("sredni blad beta: ", sr_beta / ilosc_obrazkow)

        # for i in range(ilosc_obrazkow):
        #   suma_roznic_kat += roznice_kat[i]

        # print("sredni blad: ", suma_roznice_wsp / ilosc_obrazkow)
        print("sredni blad: ", suma_roznic_kat / (ilosc_obrazkow))

X = 0
Y = 0
img = np.zeros((512, 512, 3), dtype=np.uint8)
clicked = False


def click(event, x, y, flags, param):
    global X, Y, img, clicked

    if event == cv2.EVENT_LBUTTONDOWN:
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        X, Y = x, y
        clicked = True


def robot(alfa, beta, img):
    x0 = 256
    y0 = 256
    cv2.rectangle(img, (x0 - 100, y0 + 100), (x0, y0 - 50), (100, 100, 100), -1)
    gamma = (180 - beta) / 2
    c = 100 * math.sin(math.radians(beta)) / math.sin(math.radians(gamma))
    znak_x = 1
    znak_y = 1
    if alfa <= 90 and gamma >= alfa:
        znak_x = -1 * znak_x
    if alfa >= 90 and (alfa - gamma >= 90):
        znak_y = -1 * znak_y
    x = znak_x * abs(math.sin(math.radians(abs(alfa - gamma)))) * c
    y = znak_y * abs(math.cos(math.radians(abs(alfa - gamma)))) * c
    xr = math.sin(math.radians(alfa)) * 100
    yr = math.cos(math.radians(alfa)) * 100
    cv2.circle(img, (int(x0 + xr), int(y0 - yr)), 3, (100, 100, 100), 3)
    cv2.circle(img, (int(x0 + x), int(y0 - y)), 3, (100, 100, 100), 3)
    cv2.circle(img, (X, Y), 9, (0, 255, 0), -1)
    cv2.line(img, (int(x0 + xr), int(y0 - yr)), (int(x0), int(y0)), (100, 100, 100), 3)
    cv2.line(img, (int(x0 + xr), int(y0 - yr)), (int(x0 + x), int(y0 - y)), (100, 100, 100), 3)
    for katP in range(-90, 90):
        cv2.circle(img, (x0 + int(math.cos(math.radians(katP)) * 200), y0 + int(math.sin(math.radians(katP)) * 200)), 2,
                   (0, 0, 255), -1)
    for katP in range(90, 270):
        cv2.circle(img,
                   (x0 + int(math.cos(math.radians(katP)) * 100), y0 + int(math.sin(math.radians(katP)) * 100) - 100),
                   2, (0, 0, 255), -1)


cv2.namedWindow("robot")
cv2.setMouseCallback("robot", click)
kat = [45, 90]
while True:
    if clicked:
        Xn = (X - 256 - 50) / 180
        Yn = (Y - 256) / 240
        # print(X, Y, Xn, Yn)
        for i in range(pnw):
            suma[0][i] = Xn * w1[0][i] + Yn * w1[1][i] + 1 * w1[2][i]
            # print(suma[0][i])
            wynikwarstwy[i][0] = 1 / (1 + math.exp((-1) * suma[0][i]))
            # print(wynikwarstwy[i][0])

        for i in range(pnw):
            suma[1][i] = w2[pnw][i] * 1
            for j in range(pnw):
                suma[1][i] += wynikwarstwy[j][0] * w2[j][i]
            wynikwarstwy[i][1] = 1 / (1 + math.exp((-1) * suma[1][i]))
            # print(suma[1][i])
            # print(wynikwarstwy[i][1])

        for i in range(2):
            suma[2][i] = 1 * wk[pnw][i]
            for j in range(pnw):
                suma[2][i] = wynikwarstwy[j][1] * wk[j][i]
            koniec[i] = 1 / (1 + math.exp((-1) * suma[2][i]))
            kat[i] = (koniec[i] - 0.1) * 225
            print(suma[2][i])
            print(koniec[i])
        clicked = False
        print(Xn, Yn, kat)
    robot(kat[0], kat[1], img)
    cv2.imshow("robot", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()