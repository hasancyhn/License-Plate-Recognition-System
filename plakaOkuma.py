import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# Plakanin resimdeki konumunu ogrenecegiz.
from plakaTespitEtme import plaka_konum_don
# Karakterleri taniyacagiz. a, b, c ... gibi
from plakaTanima import plakaTani
veriler = os.listdir("veriseti")

isim = veriler[3]
for isim in veriler:
    print("resim:","veriseti/"+isim)
    img = cv2.imread("veriseti/"+isim)
    img = cv2.resize(img,(500, 500))

    # x, y, w, h degerlerini application output kisminda gorecegiz.
    plaka = plaka_konum_don(img)
    plakaImg, plakaKarakter = plakaTani(img, plaka)
    print("Resimdeki Plaka:", plakaKarakter)
    plt.imshow(plakaImg)
    plt.show()
