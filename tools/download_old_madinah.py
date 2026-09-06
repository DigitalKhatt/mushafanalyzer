import os
import requests
from time import sleep

BASE_URL = "https://tafsir.app/scans/m-madinah-old"
OUT_DIR = "input/m_madinah_old"
START = 0
END = 623   # inclusive

os.makedirs(OUT_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

for i in range(START, END + 1):
    url = f"{BASE_URL}/{i}.png"
    if i >=3 and i <= 606:
        out_path = os.path.join(OUT_DIR, f"{i-2}.png")

        if os.path.exists(out_path):
            print(f"[SKIP] {i}.png already exists")
            continue

        try:
            print(f"[DOWNLOADING] {i}.png")
            r = session.get(url, timeout=15)
            r.raise_for_status()

            with open(out_path, "wb") as f:
                f.write(r.content)

            # be polite to the server
            sleep(0.1)

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] {i}.png → {e}")
