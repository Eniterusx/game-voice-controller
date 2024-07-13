# System sterowania grami za pomocą komend głosowych.

Celem projektu jest stworzenie systemu, który pozwala na sterowanie grami za pomocą komend głosowych. Działanie systemu polega na rozpoznawaniu komend głosowych, a następnie mapowaniu ich na odpowiednie klawisze klawiatury. Na przykład, jeśli użytkownik wypowie słowo *skok*, to system powinien wysłać do gry sygnał równoważny z naciśnięciem klawisza *spacja*.

## Źródła
Wykrywanie wcześniej przygotowanych komend:\
[Broadcasted Residual Learning for Efficient Keyword Spotting](https://arxiv.org/pdf/2106.04140v4)\
Autorzy: Byeonggeun Kim, Simyung Chang, Jinkyu Lee, Dooyong Sung

## Zbiór danych
### Google Speech Commands Dataset

Wersja 1: [[Train]](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz) [[Test]](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz)

Wersja 2: [[Train]](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) [[Test]](http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz)

## Rezultaty
#### Przetestowane na zbiorze danych w wersji 2
| Model (tau) | Procent zbioru danych [średnia liczba komend na klasę] | Dokładność |
|-|-|-|
| 3 | 100% [4030 komend] | 97.721% |
| 3 | 50% [2015 komend] | 97.003% |
| 3 | 30% [1209 komend] | 95.051% |
| 3 | 25% [1008 komend] | 96.141% |
| 2 | TODO | TODO |
| 6 | TODO | TODO |