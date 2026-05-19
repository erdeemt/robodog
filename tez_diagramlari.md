# Robodog — Sistem Diyagramları (Tez Eki)

Bu dosya, robotun görsel algılama ve hareket kontrol sisteminin
tüm akış, mimari ve durum diyagramlarını içerir. Diyagramlar
[Mermaid](https://mermaid.js.org/) formatında yazılmıştır.

**Görüntülemek / dışa aktarmak için:**
- [mermaid.live](https://mermaid.live) → blokları yapıştır → PNG/SVG indir
- VS Code'da **Markdown Preview Mermaid Support** eklentisi → otomatik render
- Tez Word/LaTeX dosyasına: PNG olarak export et, görsel olarak ekle


---

## 1. Genel Sistem Mimarisi

Donanım bileşenleri ve veri yolları. İki ana işlemci (Raspberry Pi 4
ve Teensy 4.1) USB-Serial ile haberleşir; her birinin sorumluluk
alanı net olarak ayrılmıştır.

```mermaid
graph LR
    subgraph PI["Raspberry Pi 4 - Görsel Algılama"]
        CAM[Camera Module 2<br/>640x480 @ 30 fps]
        DET[Kırmızı Top Dedektörü<br/>HSV + Daireselik Filtresi]
        BRG[Serial Bridge<br/>20 Hz paket]
    end

    subgraph TEENSY["Teensy 4.1 - Hareket Kontrol"]
        PARSE[CSV Parser]
        SM[Durum Makinesi<br/>SEARCH / TRACK<br/>ARRIVED / SAFE_STOP]
        PID[P-Kontrolör<br/>yaw, forward]
        IK[Ters Kinematik<br/>4 bacak]
    end

    subgraph MOTOR["Motor Sürücü Katmanı"]
        DRV[Motor Sürücüler]
        LEGS[12 x Eklem Servosu<br/>4 bacak × 3 DOF]
    end

    CAM -->|libcamera frame| DET
    DET -->|merkez, yarıçap| BRG
    BRG ==>|USB-Serial CSV<br/>F,status,dx,dy,dist,r| PARSE
    PARSE --> SM
    SM --> PID
    PID --> IK
    IK -->|PWM| DRV
    DRV --> LEGS

    classDef pi fill:#e3f2fd,stroke:#1976d2
    classDef teensy fill:#fff3e0,stroke:#e65100
    classDef motor fill:#f3e5f5,stroke:#6a1b9a
    class CAM,DET,BRG pi
    class PARSE,SM,PID,IK teensy
    class DRV,LEGS motor
```

**Tasarım kararı:** Görsel işleme Pi tarafında, kontrol mantığı Teensy
tarafında bırakılmıştır. Bu ayrım sayesinde Pi çöktüğünde Teensy
watchdog ile durur — bacaklar son komutla kontrolsüz hareket etmez.


---

## 2. Veri Akışı (Sıralı Diyagram)

Bir saniyelik tipik çalışma periyodu. 20 Hz frekansında Pi, görüş
raporunu Teensy'ye iletir; Teensy de o ana ait motor komutunu üretir.

```mermaid
sequenceDiagram
    autonumber
    participant K as Kamera
    participant D as Dedektör (Pi)
    participant B as Bridge (Pi)
    participant T as Teensy 4.1
    participant M as Motor Sürücü

    loop Her 50 ms (20 Hz)
        K->>D: Yeni frame (BGR, 640×480)
        D->>D: HSV maskele + kontur + dairesellik
        alt Top tespit edildi
            D->>B: center, radius
            B->>T: "F,1,+120,-5,52.3,28\n"
            T->>T: parse → state = TRACK
            T->>T: yaw = Kp · dx<br/>fwd = Kp · (dist - hedef)
            T->>M: drive(+0.41, +0.55)
        else Top yok
            B->>T: "F,0,0,0,0,0\n"
            T->>T: parse → state = SEARCH
            T->>M: drive(+0.40, 0)
        end
    end

    Note over T,M: 500 ms paket gelmezse → SAFE_STOP<br/>(motorlar durdurulur)
```


---

## 3. Algılama Boru Hattı (Detection Pipeline)

Bir framedeki kırmızı topun bulunması için izlenen adım dizisi.
HSV renk uzayında çift aralık maskeleme ile kırmızının açısal
sarmalı (0–10° ve 170–180°) ele alınır.

```mermaid
flowchart TD
    A([Kameradan BGR Frame]) --> B[HSV uzayına dönüştür]
    B --> C[Kırmızı maske 1<br/>H: 0–10]
    B --> D[Kırmızı maske 2<br/>H: 170–180]
    C --> E[Maskeleri birleştir<br/>bitwise OR]
    D --> E
    E --> F[Morfolojik AÇMA<br/>küçük gürültüyü sil]
    F --> G[Morfolojik KAPAMA<br/>iç boşlukları doldur]
    G --> H[Konturları bul]
    H --> I{Her kontur için}
    I --> J{Alan ≥ MIN_AREA?}
    J -->|Hayır| I
    J -->|Evet| K[Daireselik hesapla<br/>4πA / P²]
    K --> L{Dairesellik ≥ 0.6?}
    L -->|Hayır| I
    L -->|Evet| M[En büyük alanlı adayı sakla]
    I -->|Bitti| N{Aday var mı?}
    N -->|Hayır| O([None döndür])
    N -->|Evet| P[minEnclosingCircle ile<br/>merkez & yarıçap çıkar]
    P --> Q([detection: center, radius_px, area, circularity])

    style A fill:#e3f2fd
    style Q fill:#c8e6c9
    style O fill:#ffcdd2
```

**Neden iki kırmızı maske?** HSV uzayında kırmızı, açısal olarak
0° ve 180° civarında sarmalanır. Tek aralık yeterli değildir;
iki aralığın `OR`'u alınır.

**Daireselik filtresi:** Çevre uzunluğu `P` ve alan `A` için
`(4·π·A) / P²` formülü hesaplanır. Mükemmel daire için bu oran
**1.0**'dır. 0.6 eşiği topu yakalarken kırmızı kıyafet/kazak gibi
düzensiz şekilleri eler.


---

## 4. Durum Makinesi (State Machine — Teensy)

Robotun davranış durumları ve geçiş koşulları. Sistem her zaman
güvenli durumda başlar; veri akışı düzgün olduğunda diğer
durumlara geçer.

```mermaid
stateDiagram-v2
    [*] --> SAFE_STOP

    SAFE_STOP --> SEARCH    : ilk paket, status=0
    SAFE_STOP --> TRACK     : ilk paket, status=1, dist>15
    SAFE_STOP --> ARRIVED   : ilk paket, status=1, dist≤15

    SEARCH --> TRACK        : top göründü
    SEARCH --> ARRIVED      : top zaten çok yakın
    SEARCH --> SAFE_STOP    : 500 ms paket yok

    TRACK --> SEARCH        : top kayboldu
    TRACK --> ARRIVED       : dist ≤ 15 cm
    TRACK --> SAFE_STOP     : 500 ms paket yok

    ARRIVED --> SEARCH      : top kayboldu
    ARRIVED --> TRACK       : top tekrar uzaklaştı
    ARRIVED --> SAFE_STOP   : 500 ms paket yok

    SEARCH    : yerinde dön\n drive(0.4, 0)
    TRACK     : P-kontrolör\n drive(Kp·dx, Kp·(dist-25))
    ARRIVED   : dur\n drive(0, 0)
    SAFE_STOP : güvenli dur\n drive(0, 0)
```

**Watchdog mekanizması:** Pi'den son paket geleli 500 ms geçtiyse
hangi durumda olursa olsun `SAFE_STOP`'a düşülür. Bu sayede Pi
yazılımı çökse veya USB kablosu çıksa bile robot kontrolsüz hareket
etmez — yerinde durur.


---

## 5. Kontrol Karar Akışı (P-Controller)

Teensy yeni bir paket aldığında izlediği karar şeması. Çıktı:
`drive(yaw, forward)` çağrısı.

```mermaid
flowchart TD
    Start([Yeni paket alındı<br/>status, dx, dy, dist, r]) --> A{status == 0?}

    A -->|Evet — top yok| SearchBox[SEARCH<br/>yaw = +0.4<br/>fwd = 0]

    A -->|Hayır — top var| B{dist ≤ ARRIVED_CM<br/>15 cm?}

    B -->|Evet| ArrivedBox[ARRIVED<br/>yaw = 0<br/>fwd = 0]

    B -->|Hayır| C{abs dx > DEAD_BAND<br/>20 px?}

    C -->|Hayır — ortada| D[yaw = 0]
    C -->|Evet — sapmış| E[yaw = Kp_yaw · dx]

    D --> F[fwd_raw = Kp_fwd · dist - TARGET_CM]
    E --> F

    F --> G[fwd = clamp 0, MAX_FWD]

    G --> H{abs dx > %30 frame<br/>~192 px?}

    H -->|Evet — çok sapmış<br/>önce dön| I[fwd = 0]
    H -->|Hayır| J[fwd değişmez]

    I --> TrackBox[TRACK<br/>drive yaw, fwd]
    J --> TrackBox

    SearchBox --> Drive([drive motoru çağır])
    ArrivedBox --> Drive
    TrackBox --> Drive

    style Start fill:#e3f2fd
    style Drive fill:#c8e6c9
    style ArrivedBox fill:#fff9c4
    style SearchBox fill:#ffe0b2
    style TrackBox fill:#dcedc8
```

**Tasarım notu — "Önce dön" koruyucusu:** Top çok yan sapmışsa
(|dx| > frame genişliğinin %30'u), ileri komut bastırılır. Bu
sayede robot S-eğrisi çizerek topu kovalamak yerine **önce
yönelir, sonra yaklaşır** — yörünge daha verimli.


---

## 6. Mesaj Formatı (Pi → Teensy)

Pi'den Teensy'ye gönderilen her paketin yapısı:

```
F,<status>,<dx>,<dy>,<dist_cm>,<radius_px>\n
```

| Alan | Tip | Aralık | Anlam |
|---|---|---|---|
| `F` | sabit | — | Frame paketi başlığı (sync) |
| `status` | int | 0 / 1 | 0 = top yok, 1 = top var |
| `dx` | işaretli int | −320 .. +320 | Merkezin yatay sapması (px). `+` sağ |
| `dy` | işaretli int | −240 .. +240 | Merkezin dikey sapması (px). `+` aşağı |
| `dist_cm` | float | 0 / 5–500 | Tahmini mesafe (cm). Top yoksa 0 |
| `radius_px` | int | 0 / 5–200 | Tespit edilen yarıçap (kalite göstergesi) |
| `\n` | sabit | — | Paket sonu |

**Örnek paketler:**

| Paket | Anlam |
|---|---|
| `F,0,0,0,0,0\n` | Top yok |
| `F,1,+120,-5,52.3,28\n` | Top var, 120 px sağda, 52 cm uzakta |
| `F,1,0,0,18.4,75\n` | Top tam ortada, 18 cm — ARRIVED yakın |
| `F,1,-220,+10,95.0,12\n` | Top solda, 95 cm uzakta — önce dön, sonra yaklaş |

**Frekans:** 20 Hz (her 50 ms). Yapılandırılabilir (`BRIDGE_HZ`).

**Bant genişliği:** Ortalama paket ~22 byte → 22 × 20 = **440 byte/s**.
USB CDC tam-hız kanalının (12 Mbit/s = 1.5 MB/s) binde üçü.


---

## 7. Yazılım Bileşenleri Haritası

Proje dizinindeki dosyaların görev/sorumluluk haritası:

```mermaid
graph TB
    subgraph KAMERA["Pi: Algılama Katmanı"]
        CD[color_detection.py<br/>çoklu renk tespiti<br/>geliştirme/keşif aracı]
        RBD[red_ball_detector.py<br/>kırmızı top + mesafe + yön<br/>çekirdek modül]
        CC[calibrate_camera.py<br/>FOCAL_LENGTH_PX<br/>kalibrasyonu]
    end

    subgraph KOPRU["Pi: Köprü Katmanı"]
        SB[serial_bridge.py<br/>USB-Serial üzerinden<br/>20 Hz paket gönderir]
    end

    subgraph SIMUL["Test / Geliştirme"]
        TS[teensy_simulator.py<br/>donanımsız uçtan uca test<br/>state machine taklit eder]
    end

    subgraph FW["Teensy: Firmware"]
        RC[robodog_ctrl.ino<br/>parser + state machine<br/>+ P-kontrolör]
    end

    RBD -->|import| SB
    SB -.->|--dry-run pipe| TS
    SB ==>|USB CDC| RC
    CC -.->|FOCAL_LENGTH_PX| RBD

    classDef test fill:#fff9c4,stroke:#f57f17,stroke-dasharray: 5 5
    class TS,CC test
```

**Notlar:**
- `red_ball_detector.py` hem standalone çalışır (kendi `main()`'i var),
  hem de bir Python modülü olarak `serial_bridge.py` tarafından
  import edilir.
- `teensy_simulator.py` Teensy firmware'inin Python karşılığıdır —
  aynı sabit ve mantıkları kullanır. Donanım gelmeden tüm zincirin
  test edilmesini sağlar.


---

## 8. Mesafe Tahmini — Geometrik Formül

Tek kameralı sistemde topun mesafesinin tahmini, bilinen-çap
yönteminden türetilir. Kalibrasyon adımı `calibrate_camera.py`
ile odak uzaklığı (`f`, piksel cinsinden) hesaplanır.

```mermaid
flowchart LR
    A[Kalibrasyon:<br/>topu bilinen mesafede tut] --> B[Piksel çapı d_px ölç]
    B --> C["f = d_px × D_known / D_real<br/>(piksel)"]
    C --> D[.env'e FOCAL_LENGTH_PX olarak kaydet]

    E[Çalışma anı:<br/>tespit edilen yarıçap r] --> F[piksel çap = 2·r]
    F --> G["mesafe = D_real × f / 2·r<br/>(cm)"]
    G --> H[Teensy'ye dist_cm gönder]

    style A fill:#e3f2fd
    style E fill:#fff3e0
```

Formül özeti:

| Sembol | Anlam |
|---|---|
| D_real | Topun gerçek çapı (cm, sabit) |
| D_known | Kalibrasyon sırasındaki bilinen mesafe (cm) |
| d_px | O mesafedeki ölçülen piksel çap |
| f | Hesaplanan odak uzaklığı (piksel) |
| r | Çalışma anındaki yarıçap (piksel) |

**Çıkış denklemi:**

```
mesafe (cm) = (D_real × f) / (2 × r)
```


---

## 9. Sistem Çalışma Modları

```mermaid
flowchart TB
    subgraph DEV["Geliştirme Modu (Teensy yok)"]
        D1[python serial_bridge.py --dry-run]
        D2[| python teensy_simulator.py]
        D1 --> D2
    end

    subgraph CAL["Kalibrasyon Modu"]
        C1[python calibrate_camera.py<br/>topu farklı mesafelerde tut<br/>SPACE ile ölç]
        C2[ortalama FOCAL_LENGTH_PX'i .env'e yaz]
        C1 --> C2
    end

    subgraph PROD["Üretim Modu (Teensy bağlı)"]
        P1[Teensy: robodog_ctrl.ino upload]
        P2[Pi: python serial_bridge.py]
        P1 --> P2
    end

    CAL -.->|FOCAL_LENGTH_PX| PROD
    DEV -.->|davranış doğrulama| PROD

    classDef dev fill:#fff9c4
    classDef cal fill:#bbdefb
    classDef prod fill:#c8e6c9
    class D1,D2 dev
    class C1,C2 cal
    class P1,P2 prod
```


---

## Render & Export İpuçları

**Tez için yüksek çözünürlüklü PNG/SVG:**
1. [mermaid.live](https://mermaid.live) sayfasını aç
2. Diyagramın `` ```mermaid ... ``` `` bloğunu kopyala yapıştır
3. Sağ üstten **Actions → PNG/SVG** ile indir
4. SVG tercih et — tez baskısında vektörel kalır, kalitesi düşmez

**Toplu export (komut satırı, opsiyonel):**
```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i tez_diagramlari.md -o diagrams.pdf
```
