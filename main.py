import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ==============================================
# 1. Проверка и подготовка данных
# ==============================================

DATASET_PATH = "/Users/jelaletdinseytjanov/Downloads/archive/leapGestRecog/leapGestRecog"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Путь к датасету не найден: {DATASET_PATH}")

# Собираем все подпапки с изображениями
subjects = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
if not subjects:
    raise ValueError("Не найдено подпапок с данными")

# Собираем все классы жестов (папки вида 01_palm, 02_l и т.д.)
classes = []
for subject in subjects:
    subject_path = os.path.join(DATASET_PATH, subject)
    gestures = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
    classes.extend([f"{subject}/{gesture}" for gesture in gestures])

if not classes:
    raise ValueError("Не найдено папок с жестами (ожидаются папки вида 01_palm, 02_l и т.д.)")

print(f"Найдено классов жестов: {len(classes)}")
print("Первые 5 классов:", classes[:5])

# ==============================================
# 2. Сбор информации о файлах
# ==============================================

filepaths = []
labels = []

for class_dir in classes:
    full_class_path = os.path.join(DATASET_PATH, class_dir)
    try:
        class_files = [f for f in os.listdir(full_class_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not class_files:
            print(f"Предупреждение: в классе {class_dir} не найдено изображений")
            continue

        for file in class_files:
            filepaths.append(os.path.join(full_class_path, file))
            # Извлекаем название жеста из пути (например, "01_palm" из "03/01_palm")
            gesture_name = class_dir.split('/')[-1]
            labels.append(gesture_name)
    except Exception as e:
        print(f"Предупреждение: ошибка при обработке класса {class_dir}: {str(e)}")
        continue

if not filepaths:
    raise ValueError("Не найдено ни одного изображения в датасете")

# Создаем DataFrame
df = pd.DataFrame({
    'filename': filepaths,
    'class': labels
})

print(f"\nВсего изображений: {len(df)}")
print("Распределение по классам:")
print(df['class'].value_counts())

# ==============================================
# 3. Подготовка меток и разделение данных
# ==============================================

# Оставляем только первые 10 основных жестов (01_palm, 02_l, ..., 10_c)
main_gestures = [f"{i:02d}_" for i in range(1, 11)]  # 01_, 02_, ..., 10_
df = df[df['class'].str.startswith(tuple(main_gestures))]

if len(df) == 0:
    raise ValueError("После фильтрации не осталось изображений. Проверьте названия папок с жестами.")

print("\nРазмер датасета после фильтрации:", len(df))
print("Уникальные классы:", df['class'].unique())

# Преобразуем метки в числовой формат (01_palm -> 0, 02_l -> 1, и т.д.)
label_map = {gesture: i for i, gesture in enumerate(sorted(df['class'].unique()))}
df['label'] = df['class'].map(label_map)

# Разделение данных
train_df, test_df = train_test_split(
    df,
    test_size=0.15,
    stratify=df['label'],
    random_state=42
)
train_df, val_df = train_test_split(
    train_df,
    test_size=0.1765,  # 15% от исходного размера
    stratify=train_df['label'],
    random_state=42
)

print("\nРазмеры выборок:")
print(f"Обучающая: {len(train_df)}")
print(f"Валидационная: {len(val_df)}")
print(f"Тестовая: {len(test_df)}")

# ==============================================
# 4. Создание генераторов данных
# ==============================================

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1. / 255)

# Тренировочный генератор
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='label',
    target_size=(128, 128),
    color_mode='grayscale',
    class_mode='raw',
    batch_size=32,
    shuffle=True
)

# Валидационный генератор
val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col='label',
    target_size=(128, 128),
    color_mode='grayscale',
    class_mode='raw',
    batch_size=32,
    shuffle=False
)

# Тестовый генератор
test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='label',
    target_size=(128, 128),
    color_mode='grayscale',
    class_mode='raw',
    batch_size=32,
    shuffle=False
)

# ==============================================
# 5. Построение и обучение модели
# ==============================================

num_classes = len(label_map)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nАрхитектура модели:")
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]

print("\nНачало обучения...")
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=callbacks,
    steps_per_epoch=len(train_df) // 32,
    validation_steps=len(val_df) // 32
)

# ==============================================
# 6. Оценка и сохранение модели
# ==============================================

# Оценка на тестовых данных
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nРезультаты на тестовой выборке:")
print(f"Точность: {test_acc:.4f}")
print(f"Потери: {test_loss:.4f}")

# Визуализация
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Потери на обучении')
plt.plot(history.history['val_loss'], label='Потери на валидации')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.tight_layout()
plt.show()

# Матрица ошибок
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.labels

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_map.keys(),
            yticklabels=label_map.keys())
plt.xlabel('Предсказанные')
plt.ylabel('Истинные')
plt.title('Матрица ошибок')
plt.show()

# Сохранение модели
model.save("gesture_recognition_model.h5")
print("\nМодель успешно сохранена как 'gesture_recognition_model.h5'")