import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def plaka_konum_don(img):
    # Resmi iki renk uzayinda inceleyecegiz biri BGR biri gri (gray) renk formati.
    # Plakalari tanimak icin gri formatta kullaniriz.
    img_bgr = img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # islem resmi = ir_img: uzerinde islem yaptigimiz resmi temsil etmekte.

    # plakaların kenarlıkları yani siyah cercevesi 5px civarinda gezinmektedir.
    # pikselin ayirt edilebilirligi arttirmak icin gorseli iki kere blurlastirdik yani gurultuden arindirdik. Boylelikle kenarlarin belirginligi arttirildi.
    # Neden ikinci kez blurlastirdik ilk seferde ayni orandaki diger kenarlarida aldi ikinci kez yaptigimizda daha belirgin bir sonuc elde ettik.
    ir_img = cv2.medianBlur(img_gray, 5)  # 5x5
    ir_img = cv2.medianBlur(ir_img, 5)  # 5x5


    # Yogunluk merkezini alarak yani en cok tekrar eden renk pikselini alarak plakaya ulasmak yani plakanin kenarlarini vurgulamayi amacladik.
    medyan = np.median(ir_img)

    # Bu degerler sabit degerlerdir. !!!!!!!!!!!!!!!!!!!!!!!!!!!
    # alt yogunluk merkezi 2/3
    low = 0.67 * medyan
    # ust yokunluk merkezi 3/4
    high = 1.33 * medyan

    # Pikselleri low ve high arasindaki piksellerin yanindaki pikseller high'i gectiginde kenar olarak algiliyor.
    # canny algoritmasında o piksel için değer high altında kalırsa low high, üzerinde kalırsa high kabul edilir
    # Canny algoritmasindan elde ettigimiz sonuc low ve high deger arasinda kalirsa etrafindaki piksellere bakip high'in ustunde kalirsa direk olarak kenar oldugu kabul edilir.
    kenarlik = cv2.Canny(ir_img, low, high)


    # Genisletme islemi yapiliyor. Fotograftaki cizgiler yani kenarliklar daha kalin bir hale geldi.
    # np.ones((3,3),np.uint8) --> x ekseninde 3, y ekseninde 3 filtre olusturarark genisletme islemi yapiliyor.
    # iterations kac kere genisletme yapilacagini soyler burda 1 kere yapiyoruz.
    kenarlik = cv2.dilate(kenarlik, np.ones((3, 3), np.uint8), iterations=1)


    # counter işlemini plakanın kenarlarındaki 255 değerli olan beyaz pikselleri almak için kullanırız. Böylelikle plakanın resimde yeri tespit edilmiş olur
    # cv2.RETR_TREE -> hiyeralsık
    # CHAIN_APPROX_SIMPLE -> kosegenleri aldık, tum pıkseller yerine
    cnt = cv2.findContours(kenarlik, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnt[0]
    # counter'lari yani diktortgenleri, alan buyuklugune gore buyukten kucuge dogru siraladik.
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)

    H, W = 500, 500
    plaka = None

    for c in cnt:
        rect = cv2.minAreaRect(c)           # dikdortgen yapıda al (1)
        (x, y), (w, h), r = rect            # r = rotuate yani donme acisi
        # oran en az 2 (2)
        if (w > h and w > h * 2) or (h > w and h > w * 2):
            # koordinatlarini girip bir kutu olusturduk.
            box = cv2.boxPoints(rect)  # [[12,13],[25,13],[20,13],[13,45]]
            # hem tam sayi hem pozitif sayilar aliniyor
            box = np.int64(box)

            # 0 dedigi x koordinatini, 1 dedigi y koordinatini temsil ediyor.
            # minx ve miny bizim sol ust kose noktamiz oluyor
            minx = np.min(box[:, 0])
            miny = np.min(box[:, 1])
            maxx = np.max(box[:, 0])
            maxy = np.max(box[:, 1])

            # muhtemel plaka
            muh_plaka = img_gray[miny:maxy, minx:maxx].copy()
            muh_medyan = np.median(muh_plaka)   # orijinal resmi bozmamak icin kopyasini aldik.

            # kon1 = kontrolnoktasi1
            kon1 = muh_medyan > 85 and muh_medyan < 200  # yogunluk kontrolu (3)
            # birbirinin tersi seyler aslinda ayni seyleri temsil ediyorlar
            kon2 = h < 50 and w < 150  # sınır kontrolu (4)
            kon3 = w < 50 and h < 150  # sınır kontrolu (4)

            print(f"muh_plaka medyan:{muh_medyan} genislik: {w} yukseklik:{h}")
            kon = False
            # kon2 ve kon3'un ikisinden birisinin secilmesinin sebebi oranlari karistirabiliyor olmasindan dolayidir.
            if (kon1 and (kon2 or kon3)):

                # counter'in kosegenlerden cizildigi belli etmek icin 0 yaziyoruz.     # rengi BGR olarak yaziyoruz.
                # kalinligi 2 olarak yaziyoruz.
                # cv2.drawContours(img, [box], 0, (0, 0, 0), 1)                         # x,y,w,h
                plaka = [int(i) for i in [minx, miny, w, h]]
                kon = True
            else:
                # plaka değidir
                cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

            if (kon):
                return plaka

    return []

resim_adresler = os.listdir("veriseti")
img = cv2.imread("veriseti/" + resim_adresler[1])
img = cv2.resize(img, (500, 500))
plaka_konumu = plaka_konum_don(img)
print(plaka_konumu)