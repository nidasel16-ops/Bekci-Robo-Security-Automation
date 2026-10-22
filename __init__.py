import cv2
import sounddevice as sd
from scipy.io.wavfile import write
import threading
from datetime import datetime
import numpy as np
import requests
import time
import os
import queue

# --- TELEGRAM AYARLARI ---
TELEGRAM_TOKEN = "BURAYA_BOT_TOKEN_GIRILECEK"
CHAT_ID = "BURAYA_CHAT_ID_GIRILECEK"

# --- HASSASİYET AYARLARI ---
SES_ESIK_DEGERI = 0.006        # Ses algılama eşiği
SESSIZLIK_LIMITI = 2.0        # Konuşma bittikten sonra beklenecek bekleme süresi (sn)

def telegram_mesaj_gonder(mesaj):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": mesaj}, timeout=10)
    except Exception as e:
        print(f"[-] Telegram mesaj hatası: {e}")

def telegram_foto_gonder(foto_yolu, mesaj):
    def run():
        try:
            url_foto = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            with open(foto_yolu, "rb") as f:
                yanit = requests.post(url_foto, data={"chat_id": CHAT_ID, "caption": mesaj}, files={"photo": f}, timeout=15)
            
            if yanit.status_code == 200:
                print(f"[+] Fotoğraf Telegram'a iletildi.")
            else:
                print(f"[-] Telegram Foto Hatası: {yanit.text}")
                
            time.sleep(2)
            if os.path.exists(foto_yolu):
                os.remove(foto_yolu)
        except Exception as e:
            print(f"[-] Foto gönderme hatası: {e}")
            
    threading.Thread(target=run).start()

def telegram_ses_gonder(ses_yolu, mesaj):
    def run():
        try:
            url_doc = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
            with open(ses_yolu, "rb") as f:
                yanit = requests.post(
                    url_doc, 
                    data={"chat_id": CHAT_ID, "caption": mesaj}, 
                    files={"document": (os.path.basename(ses_yolu), f, "audio/wav")},
                    timeout=25
                )
                
            if yanit.status_code == 200:
                print(f"[+] Kesintisiz ses kaydı Telegram'a ulaştı!")
            else:
                print(f"[-] Telegram Ses Hatası: {yanit.text}")
                
            time.sleep(3)
            if os.path.exists(ses_yolu):
                os.remove(ses_yolu)
        except Exception as e:
            print(f"[-] Ses gönderme hatası: {e}")
            
    threading.Thread(target=run).start()

# --- KESİNTİSİZ SES AKIŞI MOTORU (STREAM) ---
ses_kuyrugu = queue.Queue()

def ses_dinleme_akisi():
    fs = 44100
    blok_boyutu = int(fs * 0.1) # 100 ms pencereler

    def audio_callback(indata, frames, time_info, status):
        ses_kuyrugu.put(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, dtype='float32', blocksize=blok_boyutu, callback=audio_callback):
        kayit_modunda = False
        tampon = []
        son_ses_zamani = 0.0

        while True:
            try:
                parca = ses_kuyrugu.get(timeout=1.0)
                rms = float(np.sqrt(np.mean(parca**2)))
                suan = time.time()

                if rms > SES_ESIK_DEGERI:
                    son_ses_zamani = suan
                    if not kayit_modunda:
                        kayit_modunda = True
                        tampon = []
                        print(f"\n[🎙️] KONUŞMA BAŞLADI (Akıcı Kayıt Devrede)...")

                if kayit_modunda:
                    tampon.append(parca)

                    # Konuşma bitti ve sessizlik süresi aşıldı mı?
                    if suan - son_ses_zamani > SESSIZLIK_LIMITI:
                        kayit_modunda = False
                        print("[✓] Konuşma bitti. Tek parça pürüzsüz ses oluşturuluyor...")

                        butun_ses = np.concatenate(tampon, axis=0)
                        # En az 1 saniyelik anlamlı bir kayıt ise gönder
                        if len(butun_ses) > fs * 1.0:
                            int_ses = np.int16(np.clip(butun_ses, -1.0, 1.0) * 32767)
                            zaman = datetime.now().strftime("%Y%m%d_%H%M%S")
                            ses_adi = f"ses_{zaman}.wav"
                            write(ses_adi, fs, int_ses)

                            sure_sn = round(len(butun_ses) / fs, 1)
                            uyari = f"🎙️ Odadaki ses kaydı tamamlandı! ({sure_sn} sn)\nTarih: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
                            telegram_ses_gonder(ses_adi, uyari)
                        tampon = []

            except queue.Empty:
                pass
            except Exception as e:
                print(f"[-] Ses işleme hatası: {e}")

# Arka planda kesintisiz akışı başlat
threading.Thread(target=ses_dinleme_akisi, daemon=True).start()

# --- KAMERA / HAREKET RADARI ---
cap = cv2.VideoCapture(0)
hareket_kilit = False

print("\n" + "="*50)
print("  👻 PROFESYONEL HAYALET MODU AKTİF")
print("  - Kesintisiz mikrofon akışı devrede (Kesilme/Pürüz yok).")
print("  - Durdurmak için: Ctrl + C")
print("="*50 + "\n")

telegram_mesaj_gonder("👻 Profesyonel Hayalet Modu Devrede!")

while True:
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    if not ret:
        break

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hareket_var = any(cv2.contourArea(c) > 5000 for c in contours)

    if hareket_var and not hareket_kilit:
        hareket_kilit = True
        zaman = datetime.now().strftime("%Y%m%d_%H%M%S")
        foto_adi = f"foto_{zaman}.jpg"

        cv2.imwrite(foto_adi, frame1)
        print(f"\n[!] HAREKET YAKALANDI! Fotoğraf iletiliyor...")

        uyari = f"🚨 DİKKAT! Odada hareket algılandı!\nTarih: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
        telegram_foto_gonder(foto_adi, uyari)

        def kilidi_ac():
            global hareket_kilit
            time.sleep(5)
            hareket_kilit = False
        threading.Thread(target=kilidi_ac, daemon=True).start()

    time.sleep(0.05)

cap.release()