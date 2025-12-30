# Отчет по второй лабораторной работе

# Постановка задачи. Цель работы.
**Цель работы:**
Получить удовольствие от работы с текстом и видеопотоком.

**Постановка задачи:**
1. Собрать или использовать готовый датасет новостей, размеченный по категориям «фейк» / «реальная новость» (например, FakeNewsNet, LIAR, Kaggle Fake News Dataset).
2. Выполнить предварительную обработку текстов.
3. Преобразовать тексты в числовое представление (векторы): TF-IDF, Bag-of-Words, Word2Vec или Sentence Transformers.
4. Обучить модель классификации
5. Оценить качество модели по метрикам: Precision, Recall, F1-score, ROC-AUC.
6. Проанализировать наиболее значимые признаки (слова или выражения), влияющие на решение модели.

****
| Вариант | Разметка | Датасет | Векторы | Модель | Группа |
| --- | --- | --- | --- | --- | --- |
| 8 | Нет | Kaggle Fake News Dataset | Предобученные | Векторы -> Umap -> Кластеризация (Kmeans, HDBSCAN) + BertTopic | 425 Миняев, Роева, Перьков, Пономарев  |

# 1 Теоретическая база

Обучение без учителя - это способ обработки данных для поиска неочевидных взаимосвязей в данных. В ходе такого обучения разделение идёт не на заданные классы, а на некоторое количество кластеров, точное количество которых неизвестно, но может быть уменьшено для большей конкретизации результатов. Такой подход требует дополнительных исследований для корректной интерпретации получаемых результатов.

``UMAP`` (Uniform Manifold Approximation and Projection) — алгоритм машинного обучения, выполняющий нелинейное снижение размерности. Применяется для преобразования многомерных данных в низкоразмерное пространство, например 2D/3D, сохраняя их топологическую структуру. 

Важнейшие параметры Umap:
- n_components (размерность проекции)
- n_neighbors (количество рассматриваемых соседей)
- min_dist (расстояние между точками данных при проекции)
- metric (способ измерения расстояния)
	- евклидово
	- манхетонское
	- чебышева
	- косинусное
	- и др.

В данной работе для Umap применяется косинусое расстояние, математически выражаемое через скалярное произведение и норму векторов:

``cos(θ) = (A · B) / (||A||·||B||)``

``HDBSCAN`` (Hierarchical Density-Based Spatial Clustering of Applications with Noise) — алгоритм кластеризации, который расширяет алгоритм DBSCAN. Основан на идее, что кластеры — это области высокой плотности, разделённые областями низкой плотности точек данных в пространстве. 

Для кластеризации через HDBSCAN применялось евклидово расстояние для двумерного случая вычисляемое как 

``d = √((x₂ - x₁)² + (y₂ - y₁)²)``

``BERTopic`` — это современный алгоритм извлечения тем из текстовых данных. При выделении определяется одна конкретная тема, к которой принадлежит передаваемая статья, что может быть недостаточно гибко в некоторых ситуациях.

# 2 Описание разработанной системы

алгоритмы, принципы работы, архитектура

Датасет представлен CSV файлом с колонками title, text, subject, date

Подготовка данных для обработки состоит из нескольких этапов:
- создание новой колонки content содержащей запись вида "title - text"
- обработка текста:
  - приведение к нижнему регистру
  - удаление ссылок
  - удаление небуквенных символов
  - приведение слов к корневой форме

В результате текст преобразовывается следующим образом:

| title - text | news["content"] |
| --- | --- |
|   Donald Trump Sends Out Embarrassing New Year’s Eve Message - This is Disturbing,"Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters and  the very dishonest fake news media.  The former reality show star had just one job to do and he couldn t do it. As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year,  President Angry Pants tweeted.  2018 will be a great year for America! As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year. 2018 will be a great year for America!  Donald J. Trump (@realDonaldTrump) December 31, 2017...| donald trump send embarrass new year eve messag disturb donald trump wish american happi new year leav instead give shout enemi hater dishonest fake news media former realiti show star one job countri rapidli grow stronger smarter want wish friend support enemi hater even dishonest fake news media happi healthi new year presid angri pant tweet great year america countri rapidli grow stronger smarter want wish friend support enemi hater even dishonest fake news media happi healthi new year great year america donald trump decemb trump tweet went welll expect kind presid send new year greet like... |

Выделение эмбедингов производится с помощью SentenceTransformer.

Полученные эмбединги обрабатываются Umap для уменьшения размерности их признаков до 5.

Далее производится выделение кластеров и оценки выбросов относительно них с использованием HDBSCAN.

Модели Umap и кластеризации используются как предобученные части сети для BERTopic, который определяет темы статей в кластерах и уменьшает их количество, объединяя схожие.

Для наглядности получаем темы с наибольшей долей выбросов:

Топ-5 тем с наибольшей долей выбросов:
| Тема | Выбросы | Ключевые слова |
| --- | --- | --- |
| 18 | 55.6% | [najib, 1mdb, malaysia, said, malaysian] |
| -1 | 44.2% | [trump, said, president, people, obama] |
| 6 | 35.0% | [duterte, philippines, drugs, police, manila] |
| 4 | 29.4% | [italy, star, malta, berlusconi, caruana] |
| 0 | 22.0% | [trump, said, president, people, state] |

Для полученных оценок несоответствия кластеру, соответствия теме, непринадлежность никакому кластеру формируется комбинированная оценка, по которой выделяются ниболее подозрительные темы. Для улучшения результата оценка должна стремиться к 0.

``0.6 * несоответствие-кластеру + 0.1 * непринадлежность-никакому-кластеру + 0.3 * (1 - соответствие-теме)``

Получаемые в процессе эмбединги, модели сохраняются на личный Google Drive для их загрузки без повторного запуска обучения.

# 3 Результаты работы и тестирования системы

Для визуального представления данных пятимерное пространство признаков после обучения с помощью Umap приводится в двумерное, результат представлен на рисунке 1, где красные точки выделяют наиболее аномальные статьи новостей.

![<img src="/graph.png">](https://github.com/MrvZzZzZ/Lab2_AI_for_study/blob/main/graph.png)

										Рисунок 1 - Проекция кластеров в двумерной плоскости

# Выводы по работе

В ходе работы изучен пример обучения нейросети без учителя для кластеризации датасета, в котором перемешаны как настоящие, так  ненастоящие новости на английском языке.
Применён способ токенизации данных для дальнейшей обработки.

Построен двумерный график с визуальным представлением кластеров, на котором помечены статьи с наибольшией вероятностью являющиеся фейком.

Для интерпретации полученных результатов требуется провести анализ получаемых кластеров для выделения какие из них относятся к фейкам, что также внесёт ясность, ккие из статей приводят к ошибкам первого рода (реальная новость относитсяя к кластеру с фейками) и второго рода (фейковая новость относится к кластеру с реальными).

# Использованные источники
1. Kaggle Fake News Dataset. https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. UMAP: Uniform Manifold Approximation and Projection. https://www.geeksforgeeks.org/machine-learning/umap-uniform-manifold-approximation-and-projection/
3. Хабр. Потрясающе красиво: как отобразить десятки признаков в данных. https://habr.com/ru/companies/skillfactory/articles/580154/
