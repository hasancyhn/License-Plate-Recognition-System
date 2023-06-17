import cv2
import numpy as np
import pickle

dosya = "rfc_model.rfc"
rfc = pickle.load(open(dosya,"rb"))
sinifs = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10,
          'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20,
          'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30,
          'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'arkaplan': 36}

index = list(sinifs.values())
siniflar = list(sinifs.keys())

def islem(img):
    yeni_boy = img.reshape((1600,5,5))
    orts = []
    for parca in yeni_boy:
        ort = np.mean(parca)
        orts.append(ort)
    orts = np.array(orts)
    orts = orts.reshape(1600,)
    return orts

def plakaAyristir(mevcutPlaka):
    mevcutPlaka = sorted(mevcutPlaka, key=lambda x:x[1])
    mevcutPlaka = np.array(mevcutPlaka)
    mevcutPlaka = mevcutPlaka[:,0]
    mevcutPlaka = mevcutPlaka.tolist()

    karakterAdim=0  # kac karakter adim attigimizi tutariz.
    for i in range(len(mevcutPlaka)):
        try:
            # plaka bir rakam ile baslamali
            int(mevcutPlaka[i])
            karakterAdim+=1
        except:
            # Karakter atlamis mi yani sayi veyahut baska bir sey yakalamis mi onu kontrol ederiz.
            # plakanin basında en fazla 2 rakam bulunabilir.
            if karakterAdim>0:
                if i-2>=0:
                    mevcutPlaka = mevcutPlaka[i-2:]

                break
            # Egerki karakter atlamamissa yani bir sey yakalayamamissa siliyoruz.
            mevcutPlaka.pop(i)

    # plaka bir sayi ile bitmeli    # plakanın sonunda en fazla 4 rakam olabilir
    karakterAdim=0
    for i in range(len(mevcutPlaka)):
        kontrolIndex = -1 + (-1*karakterAdim)
        try:
            int(mevcutPlaka[kontrolIndex])
            karakterAdim+=1
        except:
            if karakterAdim>0:
                karIndex = len(mevcutPlaka)-karakterAdim
                # print("karakter:", mevcutPlaka[karIndex])

                mevcutPlaka = mevcutPlaka[:karIndex+4]
                break
            mevcutPlaka.pop(kontrolIndex)
    
    return mevcutPlaka
def plakaTani(img,plaka):
    global index,siniflar
    x,y,w,h = plaka

    if(w>h):
        plaka_bgr = img[y:y+h,x:x+w].copy()
    else:
        plaka_bgr = img[y:y+w,x:x+h].copy()


    H,W = plaka_bgr.shape[:2]
    H,W=H*2, W*2
    plaka_bgr = cv2.resize(plaka_bgr,(W, H))

    #plaka_resim : islem resmimmiz
    plaka_resim = cv2.cvtColor(plaka_bgr, cv2.COLOR_BGR2GRAY)

    th_img = cv2.adaptiveThreshold(plaka_resim, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    th_img = cv2.morphologyEx(th_img, cv2.MORPH_OPEN,kernel, iterations=1)

    cnt = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnt[0]
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[:15]

    yaz = plaka_bgr.copy()
    mevcutPlaka = []

    for i, c in enumerate(cnt):
        rect = cv2.minAreaRect(c)  # Konturu parametre olarak verdiğimiz metot bize en küçük dikdörtgenin alanını geriye döndürecek
        # Geriye dikdörtgenin x,y kooordinatları(yoğunluk merkezi), genişlik ve yüksekliği ve açısı(rotate) döner.
        (x, y), (w,h), r = rect  # Farklı şekiller olabiliceği için yoğunluk merkezini almak karakter tespitinde daha sağlıklı olur.

        kon1 = max([w, h]) < W / 4  # Karakter genişliğimizin, ana resmimizin genişliğinin 4'te 1'inden küçük olmalıdır.
        kon2 = w * h > 200  # Karakter alanı 200'den büyük olmalıdır. Denenerek bulundu.

        if (kon1 and kon2):
            box = cv2.boxPoints(rect)  # Bulunan karakterin köşe noktalarını verir (sol üst (en kucuk x ve y) ve sağ alt köşe (en büyük x ve y)).
            box = np.int64(box)
            # Bulunan karakter köşeleri ile ilgili aşağıda bir karakterin sol üst yani min x,y koordinatından başlayarak max x,y koordinatındaki değerleri alırız
            minx = np.min(box[:, 0])
            miny = np.min(box[:, 1])
            maxx = np.max(box[:, 0])
            maxy = np.max(box[:, 1])
            # Karakteri biraz tam sınırda almak yerine 2 px daha genişletilmiş halini almamız gerekir.
            # Mesela A harfinin ucu ucuna almak yerine biraz genisletilmis halini almak daha saglikli sonuc elde etmemizi saglar.
            odak = 2
            # Aşağıda resimde kesilen karakter için min ve max koordinat değerlerini odaktana çıkarırız ve karakter daha geniş alinmis olur.
            minx = max(0, minx - odak)
            miny = max(0, miny - odak)
            maxx = min(W, maxx + odak)
            maxy = min(H, maxy + odak)
            # Karakteri plakadan kesme işlemini burada yaparız.
            kesim = plaka_bgr[miny:maxy, minx:maxx].copy()
            # Bazen plaka üzerindeki yanlış karakterleri yakaladığı için arka plan rengi sadece beyaz olan ve karakter rengi de siyah olan değerleri almayı sağladık.
            # Gri renge donusturuyoruz.
            tani = cv2.cvtColor(kesim, cv2.COLOR_BGR2GRAY)  # Burdan 0-255 arasninda bir deger gelecek.
            tani = cv2.resize(tani, (200, 200))
            # 0 veya 1 degerini alıyoruz (255 degeri beyaza denk geliyor biz 255'e bolerek 1 buluyoruz yani yine beyaz).
            tani = tani / 255
            oznitelikler = islem(tani)
            karakter = rfc.predict([oznitelikler])[0]
            ind = index.index(karakter)
            # Karakterleri ogrenmis olduk indeksler sayesinde
            sinif = siniflar[ind]
            # Indeksler disinda bir sey yakalnirsa "arkaplan" olarak algiliyoruz.
            if sinif == "arkaplan":
                continue

            cv2.putText(yaz, sinif, (minx - 2, miny - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # Ilk parametre olarak siniftaki karakter ve minx degerini aliyoruz.
            mevcutPlaka.append([sinif, minx])
            cv2.drawContours(yaz, [box], 0, (0, 255, 0), 1)
    if len(mevcutPlaka) > 0:
        mevcutPlaka = plakaAyristir(mevcutPlaka)

    return yaz, mevcutPlaka