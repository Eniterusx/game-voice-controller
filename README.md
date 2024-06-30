# System sterowania grami za pomocą komend głosowych.

Celem projektu jest stworzenie systemu, który pozwala na sterowanie grami za pomocą komend głosowych. Działanie systemu polega na rozpoznawaniu komend głosowych, a następnie mapowaniu ich na odpowiednie klawisze klawiatury. Na przykład, jeśli użytkownik powie *skok*, to system powinien wysłać do gry sygnał równoważny z naciśnięciem klawisza *spacja*.

## Źródła
Wykrywanie wcześniej przygotowanych komend:\
[Broadcasted Residual Learning for Efficient Keyword Spotting](https://arxiv.org/pdf/2106.04140v4)\
Autorzy: Byeonggeun Kim, Simyung Chang, Jinkyu Lee, Dooyong Sung

Jeśli zdecyduję się rozszerzyć model, chciałbym wprowadzić również możliwość dodawania własnych komend. Wtedy skorzystałbym również z:\
[Few-Shot Open-Set Learning for On-Device Customization of KeyWord Spotting Systems](https://arxiv.org/pdf/2306.02161v1)\
Autorzy: Manuele Rusci, Tinne Tuytelaars

## Zbiór danych
Obecnie pozyskane dane to Google Speech Commands Dataset, aczkolwiek zbiór ten będzie trzeba rozszerzyć o komendy specyficzne dla gier.

Wersja 1: [[Train]](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz) [[Test]](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz)

Wersja 2: [[Train]](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) [[Test]](http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz)