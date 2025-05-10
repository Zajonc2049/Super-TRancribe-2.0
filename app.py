# app.py для Streamlit (Повна версія з візуальним прогресом та виправленнями)

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

# --- Налаштування логування ---
# Логування на Streamlit Cloud працює дещо інакше, але ці налаштування не завадять.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# --- Шляхи до тимчасових та вихідних директорій ---
# На Streamlit Cloud ці директорії будуть тимчасовими для кожного сеансу або контейнера
BASE_DIR = os.getcwd()
TEMP_DIR = os.path.join(BASE_DIR, "temp_files")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.info(f"Створено директорії: {TEMP_DIR} та {OUTPUT_DIR}")


# --- Завантаження моделі ---

@st.cache_resource
def load_whisper_model(model_name="base"):
    """Завантажує модель Whisper і кешує її."""
    logging.info(f"Завантаження моделі Whisper: {model_name}")
    try:
        model = whisper.load_model(model_name)
        logging.info("Модель Whisper завантажено успішно.")
        return model
    except Exception as e:
        logging.error(f"Помилка під час завантаження моделі Whisper '{model_name}': {e}")
        st.error(f"Не вдалося завантажити модель Whisper: {e}") # Відображаємо помилку в UI
        return None # Повертаємо None у випадку помилки

# Завантажуємо модель при старті додатку
model = load_whisper_model("base") # Можете зробити назву моделі параметром, якщо хочете

# Перевіряємо, чи модель завантажилась успішно, перед тим як продовжувати
if model is None:
    st.stop() # Зупиняємо виконання скрипта, якщо модель не завантажилась


# --- Ваша логіка розшифровки та допоміжні функції ---
# !!! ЦЕ ВАШ КОД ФУНКЦІЙ З ПОПЕРЕДНЬОГО app.py !!!
# !!! Переконайтеся, що він скопійований повністю та правильно !!!

def transcribe_audio(audio_path, language=None, task="transcribe"):
    """
    Функція для розшифровки аудіо файлу за допомогою Whisper.
    Використовує глобальну змінну 'model'.
    """
    logging.info(f"Запуск розшифровки/перекладу для: {os.path.basename(audio_path)}, мова: {language}, завдання: {task}")
    try:
        # Використовуємо глобальну модель
        global model
        if model is None:
            raise RuntimeError("Модель Whisper не завантажена.")

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
                # Форматування часу для SRT (HH:MM:SS,ms)
                start_td = datetime.timedelta(seconds=segment['start'])
                end_td = datetime.timedelta(seconds=segment['end'])
                start_str = str(start_td).split('.', 2)[0]
                end_str = str(end_td).split('.', 2)[0]
                start_ms = int((segment['start'] - int(segment['start'])) * 1000)
                end_ms = int((segment['end'] - int(segment['end'])) * 1000)

                f.write(f"{i+1}\n{start_str},{start_ms:03d} --> {end_str},{end_ms:03d}\n{segment['text'].strip()}\n\n")
        logging.info(f"Субтитри збережено до: {srt_path}")

        logging.info("Розшифровка/переклад завершено успішно.")
        return text, srt_path, txt_path

    except Exception as e:
        logging.error(f"Помилка під час розшифровки/перекладу transcribe_audio: {e}", exc_info=True) # exc_info=True додасть повний traceback у логи
        # Повертаємо зрозуміле повідомлення про помилку та None для шляхів
        return f"Помилка під час розшифровки: {e}", None, None


# Змінена функція process_media - приймає status_object
def process_media(media_file_object, language, task, status_object):
    """
    Обробляє завантажений медіа-файл (аудіо або відео), витягує аудіо, якщо потрібно,
    та викликає функцію розшифровки.
    Приймає файловий об'єкт від Streamlit та об'єкт статусу Streamlit.
    """
    if media_file_object is None:
        status_object.update(label="Не завантажено файл", state="error")
        logging.warning("process_media викликано без файлу.")
        return "Файл не завантажено", None, None

    logging.info(f"Отримано файл для обробки: {media_file_object.name}")
    status_object.update(label=f"Отримано файл '{media_file_object.name}'", state="running", expanded=True)

    # Зберігаємо завантажений файл тимчасово...
    filename = media_file_object.name
    temp_input_path = os.path.join(TEMP_DIR, filename)

    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        status_object.update(label=f"Збереження файлу '{filename}'...", state="running")
        # Використовуємо getbuffer() для ефективного читання вмісту файлу
        with open(temp_input_path, "wb") as f:
            f.write(media_file_object.getbuffer())
        logging.info(f"Файл тимчасово збережено до: {temp_input_path}")
        status_object.write(f"Файл тимчасово збережено до: {os.path.basename(temp_input_path)}")

        ext = os.path.splitext(filename)[1].lower()
        audio_to_process_path = temp_input_path

        # Якщо це відео, витягуємо аудіо
        if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpga"]: # Додав .mpga на всяк випадок
            try:
                status_object.update(label=f"Витягування аудіо з відео ({ext})...", state="running")
                logging.info(f"Виявлено відеоформат ({ext}), витягування аудіо...")
                video = VideoFileClip(temp_input_path)
                audio_filename = os.path.splitext(filename)[0] + ".wav"
                audio_to_process_path = os.path.join(TEMP_DIR, audio_filename) # Зберігаємо витягнуте аудіо у TEMP_DIR
                logging.info(f"Витягування аудіо до: {audio_to_process_path}")
                status_object.write(f"Витягування аудіо до: {os.path.basename(audio_to_process_path)}")
                # Використовуємо render_args, щоб уникнути проблем з кодеками, якщо виникають
                video.audio.write_audiofile(audio_to_process_path, codec='pcm_s16le', verbose=False, logger=None) # Додав verbose/logger=None для тихішого виконання MoviePy
                video.close()
                logging.info("Аудіо витягнуто успішно.")
                status_object.write("Аудіо витягнуто успішно.")
            except Exception as e:
                 status_object.update(label="Помилка при витягуванні аудіо", state="error")
                 logging.error(f"Помилка при витягуванні аудіо з відео {filename}: {e}", exc_info=True)
                 # Повертаємо помилку і зупиняємо обробку для цього файлу
                 return f"Помилка при обробці відео (витягнення аудіо): {e}", None, None


        # Розшифрувати аудіо
        status_object.update(label=f"Розшифровка аудіо...", state="running")
        # Можна передати status_object в transcribe_audio, якщо вона підтримує оновлення статусу
        text_output, srt_path, txt_path = transcribe_audio(audio_to_process_path, language, task)

        # Перевіряємо, чи була помилка в transcribe_audio
        if text_output is None and (srt_path is None or txt_path is None):
             status_object.update(label="Помилка під час розшифровки", state="error")
             # Повідомлення про помилку вже повертається з transcribe_audio в text_output
             pass # Не оновлюємо статус як complete, якщо була помилка

        else:
            # Очищаємо тимчасові файли
            status_object.update(label="Очищення тимчасових файлів...", state="running")
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
                logging.info(f"Видалено тимчасовий вхідний файл: {temp_input_path}")
                status_object.write(f"Видалено тимчасовий вхідний файл: {os.path.basename(temp_input_path)}")
            # Очищаємо тимчасовий аудіо файл, якщо він був витягнутий
            if audio_to_process_path != temp_input_path and os.path.exists(audio_to_process_path):
                os.remove(audio_to_process_path)
                logging.info(f"Видалено тимчасовий аудіо файл: {audio_to_process_path}")
                status_object.write(f"Видалено тимчасовий аудіо файл: {os.path.basename(audio_to_process_path)}")

            # Вказуємо на успіх тільки якщо не було помилки в transcribe_audio
            status_object.update(label="Обробка завершена!", state="complete")

        return text_output, srt_path, txt_path

    except Exception as e:
        # Обробка будь-якої іншої помилки під час обробки медіа
        status_object.update(label=f"Виникла неочікувана помилка: {e}", state="error")
        logging.error(f"Загальна помилка в process_media: {e}", exc_info=True)
        return f"Виникла неочікувана помилка: {e}", None, None


# Функції для роботи з вихідними файлами (залишаємо ваші оригінальні)
def get_output_files():
    """Повертає список шляхів до файлів у вихідній директорії."""
    logging.info(f"Отримання списку файлів з {OUTPUT_DIR}")
    if not os.path.exists(OUTPUT_DIR):
        return []
    # Повертаємо тільки файли, ігноруючи піддиректорії
    return [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))]

def delete_all_output_files():
    """Видаляє всі файли у вихідній директорії."""
    logging.info(f"Видалення всіх файлів у {OUTPUT_DIR}")
    try:
        if os.path.exists(OUTPUT_DIR):
            for f in os.listdir(OUTPUT_DIR):
                f_path = os.path.join(OUTPUT_DIR, f)
                if os.path.isfile(f_path): # Перевіряємо, що це файл
                    os.remove(f_path)
            logging.info("Усі вихідні файли видалено успішно.")
        else:
             logging.info("Вихідна директорія не існує, видалення не потрібне.")
    except Exception as e:
        logging.error(f"Помилка при видаленні вихідних файлів: {e}", exc_info=True)
        st.error(f"Не вдалося очистити вихідні файли: {e}") # Можна додати повідомлення про помилку в UI
    return get_output_files()


# --- Побудова інтерфейсу Streamlit ---

st.title("🎤 Розшифровка аудіо та відео 🎞️")
st.markdown("Завантажте аудіо або відео файл для отримання текстової розшифровки та файлу субтитрів (SRT).")

# Використовуємо st.columns для створення колонок
col1, col2 = st.columns([2, 1]) # Створюємо 2 колонки з співвідношенням ширин

with col1:
    # st.file_uploader повертає файловий об'єкт (UploadedFile)
    uploaded_file = st.file_uploader(
    "Завантажте аудіо/відео файл",
    type=[".wav", ".mp3", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpga", ".mpeg4"], # <= ДОДАЛИ КРАПКИ ТА '.mpeg4'
    accept_multiple_files=False
)

with col2:
    # st.selectbox для мови
    language = st.selectbox(
        "Мова",
        ["auto", "uk", "en", "ru", "de", "pl", "fr", "es"], # Додайте інші мови Whisper
        index=["auto", "uk", "en", "ru", "de", "pl", "fr", "es"].index("auto")
    )
    # st.selectbox для завдання
    task = st.selectbox(
        "Завдання",
        ["transcribe", "translate"],
        index=["transcribe", "translate"].index("transcribe")
    )

# Кнопки
process_button = st.button("📝 Обробити файл")
clear_button = st.button("🔄 Очистити", help="Очистити поля вводу/виводу та вихідні файли")


# --- Логіка обробки та відображення результатів ---

# Використовуємо session_state для збереження результатів між виконаннями скрипта
if 'transcription_result' not in st.session_state:
    st.session_state.transcription_result = {"text": "", "srt_path": None, "txt_path": None}

# Перевіряємо, чи була натиснута кнопка "Обробити файл"
if process_button:
    if uploaded_file is not None:
        # Використовуємо st.status для відображення прогресу
        # expanded=True показує деталі всередині блоку статусу
        with st.status("Початок обробки...", expanded=True) as status:
            # Викликаємо вашу функцію обробки, передаючи ФАЙЛОВИЙ ОБ'ЄКТ від Streamlit ТА об'єкт статусу
            text_output_str, srt_path_result, txt_path_result = process_media(uploaded_file, language, task, status)

        # Зберігаємо результати в session state *після* того, як блок статусу завершився
        # Streamlit автоматично оновлює статус після виходу з блоку 'with'
        st.session_state.transcription_result = {
            "text": text_output_str if text_output_str is not None else "", # Гарантуємо, що текст - це рядок
            "srt_path": srt_path_result,
            "txt_path": txt_path_result
        }
        logging.info("Результати збережено в session_state.")
        # st.rerun() # Streamlit автоматично перезапускається після натискання кнопки, цей рядок не завжди потрібен, але може допомогти гарантувати оновлення

    else:
        st.warning("Будь ласка, завантажте файл перед обробкою.")

# --- Відображення результатів (читаємо з session_state) ---

st.markdown("## 📜 Результат розшифровки:")

# Отримуємо результат з стану, гарантуємо, що це рядок для st.text_area
result_text_from_state = st.session_state.transcription_result.get("text", "") # Використовуємо .get для безпеки, якщо ключ раптом відсутній
if result_text_from_state is None: # Подвійна перевірка, хоча session_state має зберігати рядок або ""
    result_text_from_state = ""

# Аналог gr.Textbox - відображаємо збережений текст
# ВИПРАВЛЕННЯ: Використовуємо disabled=True замість interactive=False
# Рядок 267 у фінальному коді може трохи відрізнятись через додавання/видалення коментарів
st.text_area(
    label="Текст",
    value=result_text_from_state, # Передаємо гарантований рядок
    height=250,
    disabled=True, # Використовуємо disabled=True для неактивного поля
    help="Розшифрований текст"
)

# Аналог gr.File для SRT та TXT - кнопки завантаження
# Перевіряємо, чи існують файли перед тим, як пропонувати завантаження
output_srt_path = st.session_state.transcription_result.get("srt_path")
output_txt_path = st.session_state.transcription_result.get("txt_path")

# Додаємо перевірку os.path.exists перед створенням кнопки завантаження
if output_srt_path and os.path.exists(output_srt_path):
    try:
        with open(output_srt_path, "rb") as f: # Читаємо як бінарний файл
            st.download_button(
                label="⬇️ Завантажити SRT файл",
                data=f,
                file_name=os.path.basename(output_srt_path),
                mime="application/x-subrip",
                key="download_srt" # Додаємо унікальний ключ
            )
    except Exception as e:
        st.error(f"Не вдалося підготувати SRT файл для завантаження: {e}")


if output_txt_path and os.path.exists(output_txt_path):
    try:
        with open(output_txt_path, "rb") as f:
            st.download_button(
                label="⬇️ Завантажити TXT файл",
                data=f,
                file_name=os.path.basename(output_txt_path),
                mime="text/plain",
                 key="download_txt" # Додаємо унікальний ключ
            )
    except Exception as e:
        st.error(f"Не вдалося підготувати TXT файл для завантаження: {e}")


# --- Логіка очищення ---

# При натисканні кнопки "Очистити"
if clear_button:
    logging.info("Натиснуто кнопку Очистити.")
    delete_all_output_files() # Видаляємо файли на сервері
    # Скидаємо стан, щоб очистити поля виводу
    st.session_state.transcription_result = {"text": "", "srt_path": None, "txt_path": None}
    logging.info("Стан додатку очищено.")
    # st.experimental_rerun() # старіший спосіб
    st.rerun() # Перезапускаємо скрипт для оновлення UI

# --- Відображення файлів на сервері (Аналог gr.Files) ---

st.markdown("## 📂 Файли на сервері:")

# Отримуємо список файлів та відображаємо кнопки для їх завантаження
current_files = get_output_files()
if current_files:
    st.write("Доступні файли:")
    # Створюємо кнопки для кожного файлу, використовуючи його шлях/ім'я як унікальний ключ
    for i, file_path in enumerate(current_files):
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                 st.download_button(
                     label=f"⬇️ {file_name}",
                     data=f,
                     file_name=file_name,
                     key=f"download_output_{i}_{file_name}" # Унікальний ключ для кожної кнопки
                 )
        except Exception as e:
             st.error(f"Не вдалося прочитати файл {file_name} для завантаження: {e}")
else:
    st.write("Наразі немає вихідних файлів на сервері.")

# Кнопка оновлення списку файлів
# Оскільки Streamlit перезапускається при багатьох діях, список оновлюватиметься.
# Можна додати окрему кнопку, якщо потрібно оновлювати список без інших дій.
# if st.button("🔄 Оновити список файлів"):
#    st.rerun()


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
# Тому я її не включаю у фінальний код.
