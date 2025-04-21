# 🧠 MRI Görüntüleri Üzerinde Tümör Tespiti için U-Net Tabanlı Derin Öğrenme Segmentasyon Modeli

## 🎯 Proje Amacı

Bu proje, beyin tümörü tespiti amacıyla U-Net mimarisi kullanarak tıbbi görüntüler üzerinde segmentasyon gerçekleştiren bir derin öğrenme modelini içermektedir.  
Eğitim süreci **PyTorch** kütüphanesi ile yapılandırılmış olup, veri setindeki her sınıfa ait (örneğin `meningioma`) görüntü ve maske çiftleri ile model eğitilmektedir.  

Eğitim sonrası model, test görüntüsü üzerinde tahmin ettiği maskeyi üretmekte ve çıktıların doğruluğu görselleştirilmektedir.  
Bu uygulama, medikal görüntü işleme ve karar destek sistemleri geliştirmek isteyen mühendisler için güçlü ve esnek bir başlangıç çerçevesi sunar.

---

## 🧠 Kullanılan Teknikler

- Derin Sinir Ağı: **U-Net Mimarisi**
- Görüntü Segmentasyonu için **Binary Cross Entropy Loss**
- **Adam Optimizer** ile parametre güncelleme
- Görüntü boyutlandırma ve dönüşüm işlemleri: `torchvision.transforms`
- Görselleştirme: `matplotlib`

---

## ⚙️ Eğitim Detayları

- Giriş görüntü boyutu: **512x512 RGB**
- Maske görüntüleri: **512x512 grayscale**
- Eğitim dönemi (epoch): `10`
- Mini-batch boyutu: `4`
- Öğrenme oranı: `0.0001`
- Optimizasyon algoritması: `Adam`
- Kayıp fonksiyonu: `BCELoss`

---

## 💾 Gereksinimler (Dependencies)

Aşağıdaki Python kütüphanelerinin kurulu olması gereklidir:

```bash
pip install torch torchvision numpy matplotlib pillow


🚀 Çalıştırma

>> python unet_segmentation.py
