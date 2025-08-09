
# Quantum Simulator App

Bu proje, **Streamlit** kullanarak temel kuantum algoritmalarını simüle eden ve görselleştiren bir eğitim uygulamasıdır. Qiskit kullanılmadan Python ve Matplotlib ile geliştirilmiştir.

## 🚀 Özellikler
- Tek qubit kapıları ile kuantum durumu simülasyonu
- Bloch küresi görselleştirmesi
- Kuantum teleportation (Bell çifti) demosu
- Adım adım kapı uygulama modu
- Docker ile kolay kurulum ve çalıştırma
- GitHub Actions ile otomatik Docker Hub yükleme

---

## Kuantum Algoritma Eğitimi Nedir?

Kuantum bilgisayarlar, klasik bilgisayarlardan çok farklı prensiplerle çalışan ve belirli problemleri çok daha hızlı çözebilen güçlü hesaplama makineleridir. Bu eğitim uygulaması, kuantum bilgisayarların nasıl çalıştığını, temel kavramlarını ve bazı önemli kuantum algoritmalarını anlamanız için tasarlanmıştır.

### Qubit Nedir?

Klasik bilgisayarlarda bilgi, **bit** denilen 0 veya 1 değerini alan en küçük birimlerle temsil edilir. Kuantum bilgisayarlarda ise bu rolü, **qubit** (kuantum biti) üstlenir. Qubitler klasik bitlerden farklı olarak;

- Aynı anda hem 0 hem de 1 durumunda bulunabilirler (**süperpozisyon**),
- İki veya daha fazla qubit arasında özel bir bağlantı (**entanglement**) kurulabilir,
- Bu özellikler sayesinde kuantum bilgisayarlar bazı problemleri çok daha hızlı çözebilir.

### Kuantum Durumu ve Bloch Küresi

Bir qubitin durumu, karmaşık bir vektörle ifade edilir. Ancak bunu anlamayı kolaylaştırmak için **Bloch küresi** adlı 3 boyutlu bir görselleştirme kullanılır. Bloch küresi, qubitin süperpozisyon ve faz bilgilerini bir küre üzerindeki noktayla gösterir.

### Kuantum Algoritmalarının Amacı Nedir?

Kuantum algoritmaları, klasik algoritmalardan farklı yöntemlerle problemleri çözmek için tasarlanmıştır. Bazı örnekler:

- **Deutsch–Jozsa Algoritması:** Bir fonksiyonun sabit mi yoksa dengeli mi olduğunu çok hızlı tespit eder. Klasik algoritmalardan çok daha hızlı sonuç verir.
- **Grover Algoritması:** Büyük veri tabanlarında hedef elemanı klasik algoritmalara göre çok daha hızlı arar.
- **Kuantum Teleportasyon:** Qubit bilgilerini fiziksel olarak taşımadan, uzaktan başka bir qubite aktarır. Bu, kuantum iletişimi için temel bir yöntemdir.

### Bu Simülatör ile Neler Öğreneceksiniz?

- Qubitlerin kuantum kapıları ile nasıl değiştiğini ve etkileşime girdiğini,
- Kuantum algoritmalarının temel mantığını ve işleyişini adım adım,
- Kuantum durumlarının görselleştirilmesini (Bloch küresi),
- Kuantum teleportasyonun çalışma prensibini,
- Kuantum hesaplamanın klasik hesaplamadan farklarını simülasyonla deneyimleyerek öğreneceksiniz.

### Kimler İçin?

- Kuantum bilgisayar ve kuantum algoritmaları hakkında temel ve uygulamalı bilgi edinmek isteyenler,
- Fizik veya matematik alanında derin bilgiye sahip olmayan ama kuantuma meraklı olanlar,
- Programlama yoluyla kuantum kavramlarını keşfetmek isteyen öğrenciler ve araştırmacılar,
- Kuantum teknolojilerinin geleceğini anlamak isteyen teknoloji meraklıları.

### Sonuç

Bu uygulama, kuantum dünyasına adım atmak isteyen herkese açık ve anlaşılır bir başlangıç noktasıdır. Kodları çalıştırarak, görselleştirerek ve deneyimleyerek kuantum hesaplamanın temelini kavrayabilirsiniz.

---

## 📦 Kurulum

### Yerel Çalıştırma
```bash
pip install -r requirements.txt
python generate_images.py
streamlit run streamlit_quantum_simulator_app_extended.py
```

### Docker ile Çalıştırma
```bash
docker pull <dockerhub_kullanici_adiniz>/quantum-simulator:latest
docker run -p 8501:8501 <dockerhub_kullanici_adiniz>/quantum-simulator:latest
```

---

## 📊 Örnek Görseller
### Bloch Küresi
![Bloch Sphere](images/bloch_sphere_example.png)

### Kuantum Teleportation Devresi
![Teleportation Circuit](images/teleportation_circuit_example.png)

> **Not:** `images` klasöründe dosyalar bulunmuyorsa `python generate_images.py` komutunu çalıştırın.

---

## 📜 Proje Yapısı
```
.
├── streamlit_quantum_simulator_app_extended.py
├── requirements.txt
├── Dockerfile
├── README.md
├── generate_images.py
├── images/
└── .github/workflows/docker-build.yml
```

---

## ⚙️ GitHub Actions
**main** branch'e yapılan her push sonrası Docker imajı build edilip Docker Hub'a yüklenir.

Secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_PASSWORD` veya `DOCKERHUB_TOKEN`

---

## 📚 Kuantum Bilgisayar Kavramları

**Bloch Küresi:** Qubit’in kuantum durumunu 3B küre üzerinde gösteren görselleştirme yöntemi.

$$
|\psiangle = \cos\left(rac{	heta}{2}ight)|0angle + e^{i\phi} \sin\left(rac{	heta}{2}ight)|1angle
$$

Burada:

- \(	heta\) → kutupsal açı
- \(\phi\) → faz açısı

**Kuantum Teleportation:** Bilinmeyen bir kuantum durumunun fiziksel aktarım yapılmadan başka bir qubit’e iletilmesini sağlayan protokol.

---

## 👨‍💻 Katkıda Bulunma
1. Fork yapın  
2. Yeni bir branch oluşturun (`git checkout -b feature/yenilik`)  
3. Değişikliklerinizi commit edin  
4. Branch’i push edip Pull Request açın  

---

## 📄 Lisans
MIT Lisansı
