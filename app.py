# app.py для Streamlit

import streamlit as st
import whisper
import os
import shutil
import datetime
import time
import threading
import logging
import sys
from moviepy.editor import VideoFileClip # Використовується для обробки відео
import io # Потрібно для читання завантажених файлів

# --- Налаштування та директорії ---

# Налаштування логування (можна залишити або адаптувати для Streamlit Cloud)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# Шляхи до тимчасових та вихідних директорій
# На Streamlit Cloud ці директорії будуть тимчасовими для кожного сеансу або контейнера
BASE_DIR = os.getcwd()
TEMP_DIR = os.path.join(BASE_DIR, "temp_files")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Завантаження моделі ---

@st.cache_resource
def load_whisper_model(model_name="base"):
    """Завантажує модель Whisper і кешує її."""
    logging.info(f"Завантаження моделі Whisper: {model_name}")
    model = whisper.load_model(model_name)
    logging.info("Модель Whisper завантажено.")
    return model

# Завантажуємо модель при старті додатку
model = load_whisper_model("base")

# --- Ваша логіка розшифровки та допоміжні функції ---
# !!! УВАГА: Скопіюйте сюди повний код функцій з вашого попереднього app.py !!!

def transcribe_audio(audio_path, language=None, task="transcribe"):
    """
    Функція для розшифровки аудіо файлу за допомогою Whisper.
    Використовує глобальну змінну 'model'.
    """
    logging.info(f"Запуск розшифровки/перекладу для: {audio_path}, мова: {language}, завдання: {task}")
    try:
        result = model.transcribe(audio_path, language=language if language != "auto" else None, task=task)
        text = result["text"]

        # Створення шляхів для вихідних файлів
        base_filename = os.path.basename(audio_path).rsplit('.', 1)[0] # Отримуємо ім'я без розширення
        txt_path = os.path.join(OUTPUT_DIR, base_filename + ".txt")
        srt_path = os.path.join(OUTPUT_DIR, base_filename + ".srt")

        # Переконайтеся, що OUTPUT_DIR існує перед записом
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Збереження тексту в TXT
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"Текст збережено до: {txt_path}")

        # Збереження субтитрів у SRT
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"]):
                start = str(datetime.timedelta(seconds=segment['start'])).split('.', 2)[0] # Формат HH:MM:SS
                end = str(datetime.timedelta(seconds=segment['end'])).split('.', 2)[0]   # Формат HH:MM:SS
                start_ms = int((segment['start'] - int(segment['start'])) * 1000) # Мілісекунди
                end_ms = int((segment['end'] - int(segment['end'])) * 1000)     # Мілісекунди
                f.write(f"{i+1}\n{start},{start_ms:03d} --> {end},{end_ms:03d}\n{segment['text'].strip()}\n\n")
        logging.info(f"Субтитри збережено до: {srt_path}")

        logging.info("Розшифровка/переклад завершено успішно.")
        return text, srt_path, txt_path

    except Exception as e:
        logging.error(f"Помилка під час розшифровки/перекладу: {e}")
        return f"Помилка під час обробки: {e}", None, None


def process_media(media_file_object, language, task):
    """
    Обробляє завантажений медіа-файл (аудіо або відео), витягує аудіо, якщо потрібно,
    та викликає функцію розшифровки.
    Приймає файловий об'єкт від Streamlit.
    """
    if media_file_object is None:
        logging.warning("process_media викликано без файлу.")
        return "Файл не завантажено", None, None

    logging.info(f"Отримано файл для обробки: {media_file_object.name}")

    # Зберігаємо завантажений файл тимчасово, оскільки Whisper та VideoFileClip
    # часто потребують шлях до файлу, а не байтовий потік.
    filename = media_file_object.name
    temp_input_path = os.path.join(TEMP_DIR, filename)

    try:
        # Переконайтеся, що TEMP_DIR існує перед записом
        os.makedirs(TEMP_DIR, exist_ok=True)
        with open(temp_input_path, "wb") as f:
            f.write(media_file_object.getbuffer()) # Записуємо вміст файлового об'єкта

        logging.info(f"Файл тимчасово збережено до: {temp_input_path}")

        ext = os.path.splitext(filename)[1].lower()
        audio_to_process_path = temp_input_path # Спочатку вважаємо, що аудіо це сам файл

        # Якщо це відео, витягуємо аудіо
        if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            try:
                logging.info(f"Виявлено відеоформат ({ext}), витягування аудіо...")
                video = VideoFileClip(temp_input_path)
                audio_filename = os.path.splitext(filename)[0] + ".wav"
                audio_to_process_path = os.path.join(TEMP_DIR, audio_filename) # Зберігаємо витягнуте аудіо у TEMP_DIR
                logging.info(f"Витягування аудіо до: {audio_to_process_path}")
                video.audio.write_audiofile(audio_to_process_path, codec='pcm_s16le')
                video.close()
                logging.info("Аудіо витягнуто успішно.")
            except Exception as e:
                logging.error(f"Помилка при витягуванні аудіо з відео {filename}: {e}")
                # Можливо, спробувати розшифрувати відеофайл напряму, якщо витягти аудіо не вдалося?
                # Або просто повернути помилку:
                return f"Помилка при обробці відео: {e}", None, None

        # Викликаємо вашу функцію розшифровки з шляхом до аудіо (або витягнутого аудіо)
        text_output, srt_path, txt_path = transcribe_audio(audio_to_process_path, language, task)

        # Очищаємо тимчасовий вхідний файл після обробки
        if os.path.exists(temp_input_path):
             os.remove(temp_input_path)
             logging.info(f"Видалено тимчасовий вхідний файл: {temp_input_path}")
        # Очищаємо тимчасовий аудіо файл, якщо він був витягнутий з відео
        if audio_to_process_path != temp_input_path and os.path.exists(audio_to_process_path):
             os.remove(audio_to_process_path)
             logging.info(f"Видалено тимчасовий аудіо файл: {audio_to_process_path}")


        return text_output, srt_path, txt_path

    except Exception as e:
        logging.error(f"Загальна помилка в process_media: {e}")
        return f"Виникла помилка: {e}", None, None


# Функції для роботи з вихідними файлами
def get_output_files():
    """Повертає список шляхів до файлів у вихідній директорії."""
    logging.info(f"Отримання списку файлів з {OUTPUT_DIR}")
    if not os.path.exists(OUTPUT_DIR):
        return []
    return [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))]

def delete_all_output_files():
    """Видаляє всі файли у вихідній директорії."""
    logging.info(f"Видалення всіх файлів у {OUTPUT_DIR}")
    try:
        if os.path.exists(OUTPUT_DIR):
            for f in os.listdir(OUTPUT_DIR):
                f_path = os.path.join(OUTPUT_DIR, f)
                if os.path.isfile(f_path):
                    os.remove(f_path)
            logging.info("Усі вихідні файли видалено успішно.")
        else:
             logging.info("Вихідна директорія не існує, видалення не потрібне.")
    except Exception as e:
        logging.error(f"Помилка при видаленні вихідних файлів: {e}")
    # Повертаємо порожній список або оновлений список
    return get_output_files()


# Функція clear_inputs та auto_cleanup_temp_dirs - адаптуємо або прибираємо.
# Логіка очищення інтерфейсу в Streamlit інша (керується станом або перезапуском скрипта)
# Автоматичне очищення файлів у фоновому потоці може не працювати стабільно на Streamlit Cloud.
# Для початку, покладемося на кнопку "Очистити" та ефемерність середовища.
# Якщо потрібне автоматичне очищення, це вимагає більш складних рішень.


# --- Побудова інтерфейсу Streamlit ---

st.title("🎤 Розшифровка аудіо та відео 🎞️")
st.markdown("Завантажте аудіо або відео файл для отримання текстової розшифровки та файлу субтитрів (SRT).")

# Використовуємо st.columns для створення колонок, аналогічно gr.Row та gr.Column
col1, col2 = st.columns([2, 1]) # Створюємо 2 колонки з співвідношенням ширин

with col1:
    # Аналог gr.File(label="Завантажте аудіо/відео файл", type="filepath")
    # Використовуємо type='auto' (або bytes), оскільки type='filepath' може не працювати
    # на всіх платформах. Будемо зберігати файл вручну в process_media.
    uploaded_file = st.file_uploader(
        "Завантажте аудіо/відео файл",
        type=["wav", "mp3", "mp4", "mov", "avi", "mkv", "webm", "mpga"], # Додайте всі типи, які підтримуєте
        accept_multiple_files=False # Дозволяємо лише один файл
    )

with col2:
    # Аналог gr.Dropdown для мови
    language = st.selectbox(
        "Мова",
        ["auto", "uk", "en", "ru", "de", "pl", "fr", "es"], # Додайте інші мови, якщо підтримуєте Whisper
        index=["auto", "uk", "en", "ru", "de", "pl", "fr", "es"].index("auto") # Встановлюємо значення за замовчуванням
    )
    # Аналог gr.Dropdown для завдання
    task = st.selectbox(
        "Завдання",
        ["transcribe", "translate"],
        index=["transcribe", "translate"].index("transcribe") # Встановлюємо значення за замовчуванням
    )

# Кнопки
process_button = st.button("📝 Обробити файл")
clear_button = st.button("🔄 Очистити", help="Очистити поля вводу/виводу та вихідні файли")


# --- Логіка обробки після натискання кнопки ---

# Використовуємо session_state для збереження результатів між виконаннями скрипта
if 'transcription_result' not in st.session_state:
    st.session_state.transcription_result = {"text": "", "srt_path": None, "txt_path": None}

# Перевіряємо, чи була натиснута кнопка "Обробити файл" І чи завантажено файл
if process_button:
    if uploaded_file is not None:
        # Додаємо спіннер або повідомлення, поки йде обробка
        with st.spinner(f"Обробка файлу '{uploaded_file.name}'... Це може зайняти деякий час."):
            # Викликаємо вашу функцію обробки, передаючи файловий об'єкт від Streamlit
            text_output_str, srt_path_result, txt_path_result = process_media(uploaded_file, language, task)

            # Зберігаємо результати в session_state
            st.session_state.transcription_result = {
                "text": text_output_str,
                "srt_path": srt_path_result,
                "txt_path": txt_path_result
            }
            logging.info("Результати збережено в session_state.")
            # Streamlit автоматично перезапустить скрипт для оновлення UI
            st.rerun() # Явно перезапускаємо для гарантованого оновлення

    else:
        st.warning("Будь ласка, завантажте файл перед обробкою.")

# --- Відображення результатів (читаємо з session_state) ---

st.markdown("## 📜 Результат розшифровки:")

# Аналог gr.Textbox - відображаємо збережений текст
st.text_area(
    label="Текст",
    value=st.session_state.transcription_result["text"],
    height=250,
    interactive=False, # Користувач не може редагувати результат
    help="Розшифрований текст"
)

# Аналог gr.File для SRT та TXT - кнопки завантаження
# Перевіряємо, чи існують файли перед тим, як пропонувати завантаження
output_srt_path = st.session_state.transcription_result["srt_path"]
output_txt_path = st.session_state.transcription_result["txt_path"]

if output_srt_path and os.path.exists(output_srt_path):
    with open(output_srt_path, "rb") as f: # Читаємо як бінарний файл
        st.download_button(
            label="⬇️ Завантажити SRT файл",
            data=f,
            file_name=os.path.basename(output_srt_path),
            mime="application/x-subrip"
        )
if output_txt_path and os.path.exists(output_txt_path):
    with open(output_txt_path, "rb") as f:
        st.download_button(
            label="⬇️ Завантажити TXT файл",
            data=f,
            file_name=os.path.basename(output_txt_path),
            mime="text/plain"
        )

# --- Логіка очищення ---

# При натисканні кнопки "Очистити"
if clear_button:
    delete_all_output_files() # Видаляємо файли на сервері
    # Скидаємо стан, щоб очистити поля виводу
    st.session_state.transcription_result = {"text": "", "srt_path": None, "txt_path": None}
    logging.info("Стан додатку очищено.")
    st.rerun() # Перезапускаємо скрипт для оновлення UI

# --- Відображення файлів на сервері (Аналог gr.Files) ---

st.markdown("## 📂 Файли на сервері:")

# Отримуємо список файлів та відображаємо кнопки для їх завантаження
current_files = get_output_files()
if current_files:
    st.write("Доступні файли:")
    for file_path in current_files:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                 st.download_button(
                     label=f"⬇️ {file_name}",
                     data=f,
                     file_name=file_name,
                     key=f"download_{file_name}" # Унікальний ключ для кожної кнопки
                 )
        except Exception as e:
             st.error(f"Не вдалося прочитати файл {file_name}: {e}")
else:
    st.write("Наразі немає вихідних файлів на сервері.")

# Кнопка оновлення списку файлів
# Оскільки Streamlit перезапускається при багатьох діях, список оновлюватиметься.
# Можна додати окрему кнопку, якщо потрібно оновлювати список без інших дій.
# if st.button("🔄 Оновити список файлів"):
#    st.rerun() # Просто перезапускаємо, щоб оновити список

# --- Footer ---
st.markdown(
    """
    <div style="text-align: center; margin-top: 40px; color: grey; font-size: 0.9em;">
        Розроблено з використанням Whisper та Streamlit
    </div>
    """,
    unsafe_allow_html=True # Дозволяє вставляти HTML
)

# --- Примітки щодо фонового очищення ---
# Ваша функція auto_cleanup_temp_dirs та пов'язаний з нею потік
# можуть не працювати надійно в середовищі Streamlit Cloud.
# Краще покластися на механізм кнопки "Очистити" та на те, що тимчасові файли
# зникнуть після завершення сесії або контейнера.
# Якщо автоматичне очищення за часом критично важливе, це вимагатиме
# більш складних підходів, можливо, з використанням окремих сервісів або іншої архітектури.
# Тому я не включаю її сюди у фінальний код для Streamlit.
