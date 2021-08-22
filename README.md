# Sıfırdan Yapay Sinir Ağı Modeli

Hazır kütüphane kullanmadan yapay sinir ağı modeli oluşturarak opencv ve mediapipe kütüphaneleri kullanılarak elde edilmiş olan insan postürüne ait el konum verilerinden sayı tahmini yapmak.

Bağımlılıklar
- Numpy
- Pandas
- Matplotlib
- OpenCv (Sadece örnek projede veri toplanması ve test edilmesi için)
- Mediapipe (Sadece örnek projede veri toplanması ve test edilmesi için)

![ysagif](https://user-images.githubusercontent.com/67061626/130355986-e9eb4f63-c5c9-420d-9cf7-f2775e450b8d.gif)

Örnek Kullanım Training için:

```python
from training import train
layers = [[42], [1200, "relu"], [600, "sigmoid"], [5, "softmax"]]
net = Net(layers)

data =pandas.read_csv(FILEPATH)
input_data = data.iloc[:,3:45].to_numpy()
output_data = data.iloc[:,45:50].to_numpy()

data = pandas.read_csv(FILEPATH)
validation_data_input = data.iloc[:,3:45].to_numpy()
validation_data_output = data.iloc[:,45:50].to_numpy()

net.train(input_data, output_data, validation_data_input, validation_data_output, 50, 32, learningRate=0.0003,regularization=("None",0), shuffledMode=True, optimizer="None")
```

net.train(training_input_data, training_output_data, validation_data_input, validation_data_output, epoch, minibatch, learningRate, regularization, shuffledMode, optimizer)

regularization = (Regularization_Name, Regularization_Rate)   # Regularizasyon Seçenekleri "L1" ve "L2"
shuffledMode: True ya da False  # Eğitim sırasında input ve output verisinin her epok arasında karıştırılmasını sağlar
optimizer = Optimizer_Name  # Optimizasyon Seçenekleri "momentum", "rmsprob" ve "adam"


Test Verisi İçin Örnek Kullanım:

```python
data = pandas.read_csv(FILEPATH)
test_data_input = data.iloc[:, 3:45].to_numpy()
test_data_output = data.iloc[:, 45:50].to_numpy()
net.test(test_data_input,test_data_output)
```

![example1](https://user-images.githubusercontent.com/67061626/130355938-3ce61d1b-d433-4537-bee3-2e6573497c0a.png) ![example2](https://user-images.githubusercontent.com/67061626/130355972-8eb18692-1a22-47a8-b3ec-173f79804810.png)





Eklenenler
- ELUP, RELU, TANH, Sigmoid, Softmax Fonksiyonları
- Regularization tek komut ile tüm katmanlara uygulanabiliyor
- Momentum, RMSProb ve Adam optimizer'ler için kolay kullanılabilir tek fonksiyon
- Matplotlib ile Training ve Validation dataları için hata ve başarım grafikleri için görselleştirme seçeneği
- Parametreleri kaydetmek ve eğitilmiş modeli kullanabilmek için seçenek


Eksikler
- ELUP ve TANH Fonksiyonları Bellekte Aşıma Sebep Oluyor
- Katman Sayısının Değiştirilememesi
- İlk rastgele parametre atamaları için seçilebilir randomize metodu (uniform,normal dağılım vs)
- One Hot Encoding Eklentisi
- Dropout 
