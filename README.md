<img width="1173" height="634" alt="screenshot" src="https://github.com/user-attachments/assets/2257982a-9cae-41ab-8c76-45856ca11905" />


# PhysLab AI — Sönümlü & Zorlamalı Harmonik Osilatör

Sönümlü ve zorlamalı bir harmonik osilatörü sayısal olarak çözen, sonucu analitik çözümle karşılaştırarak **doğrulayan** interaktif bir web uygulaması.


İlk sürümü üretken yapay zeka (Gemini) ile oluşturuldu. Sonra kod fiziksel ve sayısal hatalar açısından incelendi, test edildi ve düzeltildi. Bu süreç aşağıda "Geliştirme süreci" bölümünde anlatılıyor.

![ekran görüntüsü](screenshot.png)

## Fizik

Hareket denklemi:

```math
m\ddot{x} + b\dot{x} + kx = F_0\cos(\omega_d t), \qquad x(0)=0,\; \dot{x}(0)=0
```

Kararlı hal çözümü x_p(t) = A cos(ω_d t − δ) denklemde yerine konunca:

```math
A = \frac{F_0}{\sqrt{(k - m\omega_d^2)^2 + (b\,\omega_d)^2}}, \qquad \tan\delta = \frac{b\,\omega_d}{k - m\omega_d^2}
```

Bir periyot üzerinden ortalama mekanik enerji:

```math
\langle E\rangle = \frac{A^2}{4}\left(k + m\omega_d^2\right)
```

Geçici (homojen) çözüm e^(−t/τ) ile söner, burada τ = 2m/b.

## Sayısal yöntem

- **Çözücü:** `scipy.integrate.solve_ivp` (RK45, `rtol = atol = 1e-8`)
- **Simülasyon süresi:** t_end = 7τ + 10·T_d. Ölçüm penceresi 7τ anında başlar; o anda geçici rejimden kalan pay e⁻⁷ ≈ %0.09'dur.
- **Ölçüm penceresi:** tam 10 sürücü periyodu. Böylece enerji ortalaması ⟨cos²⟩ = ½ koşulunu sağlar.
- **Örnekleme:** periyot başına 80 nokta. Tepe noktasını ıskalamaktan gelen en kötü genlik hatası 1 − cos(π/80) ≈ %0.08'dir.

## Doğrulama

Analitik ve sayısal kararlı hal değerlerinin karşılaştırması (F₀ = 1 N):

| m (kg) | k (N/m) | b (kg/s) | ω_d (rad/s) | τ (s) | A analitik (m) | A sayısal (m) | Hata | ⟨E⟩ analitik (J) | ⟨E⟩ sayısal (J) | Hata |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.25 | 1.8 | 8 | 1.1322 | 1.1322 | %0.003 | 2.3202 | 2.3199 | %0.013 |
| 1 | 4 | 0.3 | 2.8 | 6.7 | 0.2544 | 0.2545 | %0.051 | 0.1916 | 0.1916 | %0.000 |
| 1 | 4 | 0.3 | 2.0 (rezonans) | 6.7 | 1.6667 | 1.6665 | %0.010 | 5.5556 | 5.5535 | %0.038 |
| 5 | 4 | 0.05 | 0.9 | 200 | 14.8659 | 14.8681 | %0.015 | 444.75 | 444.85 | %0.023 |
| 1 | 4 | 0.3 | 0.2 | 6.7 | 0.2525 | 0.2527 | %0.079 | 0.0644 | 0.0644 | %0.000 |
| 5 | 4 | 0.05 | 5.0 | 200 | 0.0083 | 0.0083 | %0.072 | 0.0022 | 0.0022 | %0.000 |

Bütün durumlarda hata %0.1'in altında. Kalan küçük farklar beklenen iki kaynaktan geliyor: geçici rejimin artığı (e⁻⁷) ve örnekleme (1 − cos(π/80)).

## Geliştirme süreci: yapay zekanın ilk sürümünde bulunan hatalar

| Sorun | Etkisi | Düzeltme |
|---|---|---|
| Başlangıç koşulu x₀ = 1 m "maksimum genliğe" dahildi | Genlik gerçeğinin 4 katı gösteriliyordu (0.25 m yerine 1.016 m) | x₀ = 0 ve genlik sadece kararlı halden ölçülüyor |
| Ortalama enerji geçici rejimi de içeriyordu | 0.19 J yerine 0.72 J | Ölçüm 7τ'dan sonra yapılıyor |
| Sabit 20 s simülasyon süresi | τ = 200 s için hata %97'ye çıkıyordu | Süre fiziğe göre ayarlanıyor: 7τ + 10·T_d |
| Ölçüm penceresi tam sayıda periyot içermiyordu | Düşük frekansta enerji hatası %38 | Pencere tam 10 periyot |
| Sunucu daemon thread'de çalışıyordu | `python physlab.py` ile çalıştırınca sunucu hemen kapanıyordu | `if __name__ == "__main__"` + ana thread'de `uvicorn.run` |
| Slider'ın her adımı yeni istek atıyordu | Arayüz takılıyordu | 200 ms debounce + `AbortController` |
| Uzun simülasyonlarda grafik periyot başına 1 noktadan az alıyordu | Aliasing (yanlış dalga deseni) | Son 5 periyot yakın çekim + tüm süre için zarf grafiği |

## Kurulum ve çalıştırma

```bash
pip install -r requirements.txt
python physlab.py
```

Tarayıcı otomatik açılır (port 8000 doluysa boş bir port seçilir). Alternatif olarak:

```bash
uvicorn physlab:app --reload
```

API'yi doğrudan da çağırabilirsin:

```
http://127.0.0.1:8000/sim?m=1&k=4&g=0.25&w=1.8
```

## Kullanılan teknolojiler

Python · FastAPI · NumPy · SciPy · Chart.js · Tailwind CSS
