# app.py для Streamlit (Інтегрована версія з розшифровкою та перекладом)

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
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import tempfile

# --- Налаштування логування ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# --- Шляхи до тимчасових та вихідних директорій ---
BASE_DIR = os.getcwd()
TEMP_DIR = os.path.join(BASE_DIR, "temp_files")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.info(f"Створено директорії: {TEMP_DIR} та {OUTPUT_DIR}")


# --- ФУНКЦІЇ ЗАВАНТАЖЕННЯ МОДЕЛЕЙ ---

# Завантаження моделі Whisper
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
        st.error(f"Не вдалося завантажити модель Whisper: {e}")
        return None

# Завантаження моделі M2M-100 для перекладу
@st.cache_resource
def load_m2m100_model(model_size="facebook/m2m100_418M"):
    """
    Завантажує модель M2M-100 та токенізатор
    Доступні розміри: 
    - facebook/m2m100_418M (менша версія ~1GB)
    - facebook/m2m100_1.2B (більша версія ~2.4GB)
    """
    logging.info(f"Завантаження моделі M2M-100: {model_size}")
    try:
        tokenizer = M2M100Tokenizer.from_pretrained(model_size)
        model = M2M100ForConditionalGeneration.from_pretrained(model_size)
        logging.info("Модель M2M-100 завантажено успішно.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Помилка завантаження моделі M2M-100: {e}")
        st.error(f"Помилка завантаження моделі M2M-100: {e}")
        return None, None

# Завантажуємо модель при старті додатку
whisper_model = load_whisper_model("base")

# Перевіряємо, чи модель завантажилась успішно, перед тим як продовжувати
if whisper_model is None:
    st.stop()


# --- ФУНКЦІЇ ОБРОБКИ ---

def transcribe_audio(audio_path, language=None, task="transcribe"):
    """
    Функція для розшифровки аудіо файлу за допомогою Whisper.
    Використовує глобальну змінну 'whisper_model'.
    """
    logging.info(f"Запуск розшифровки/перекладу для: {os.path.basename(audio_path)}, мова: {language}, завдання: {task}")
    try:
        # Використовуємо глобальну модель
        global whisper_model
        if whisper_model is None:
            raise RuntimeError("Модель Whisper не завантажена.")

        result = whisper_model.transcribe(audio_path, language=language if language != "auto" else None, task=task)
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
        return text, srt_path, txt_path, result["segments"]

    except Exception as e:
        logging.error(f"Помилка під час розшифровки/перекладу transcribe_audio: {e}", exc_info=True)
        # Повертаємо зрозуміле повідомлення про помилку та None для шляхів
        return f"Помилка під час розшифровки: {e}", None, None, None


# Функція для перекладу тексту за допомогою M2M-100
def translate_with_m2m100(text, source_lang, target_lang):
    """
    Перекладає текст з вихідної мови на цільову за допомогою моделі M2M-100.
    """
    logging.info(f"Запуск перекладу тексту з {source_lang} на {target_lang}")
    model, tokenizer = load_m2m100_model()
    
    if not model or not tokenizer:
        return "Не вдалося завантажити модель перекладу"
    
    try:
        # Встановлюємо вихідну мову для токенізатора
        tokenizer.src_lang = source_lang
        
        # Токенізуємо вхідний текст
        encoded_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Генеруємо переклад
        with torch.no_grad():
            # Встановлюємо токен цільової мови для генерації
            generated_tokens = model.generate(
                **encoded_text,
                forced_bos_token_id=tokenizer.get_lang_id(target_lang),
                max_length=512
            )
        
        # Декодуємо результат
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        # Збереження перекладу в TXT файл
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        translation_filename = f"translation_{source_lang}_to_{target_lang}_{timestamp}.txt"
        translation_path = os.path.join(OUTPUT_DIR, translation_filename)
        
        with open(translation_path, "w", encoding="utf-8") as f:
            f.write(translated_text)
        
        logging.info(f"Переклад успішно збережено до: {translation_path}")
        return translated_text, translation_path
    
    except Exception as e:
        error_msg = f"Помилка під час перекладу: {e}"
        logging.error(error_msg, exc_info=True)
        return error_msg, None


# Функція для отримання підтримуваних мов M2M-100
def get_supported_languages():
    """Повертає словник підтримуваних мов M2M-100"""
    return {
        "uk": "українська",
        "en": "англійська",
        "ru": "російська",
        "de": "німецька",
        "fr": "французька",
        "es": "іспанська",
        "pl": "польська",
        "it": "італійська",
        "cs": "чеська",
        "ja": "японська",
        "zh": "китайська",
        "ko": "корейська",
        "ar": "арабська",
        "tr": "турецька",
        "vi": "в'єтнамська",
        "pt": "португальська",
        "be": "білоруська",
        "sk": "словацька",
        "bg": "болгарська",
        "nl": "нідерландська", 
        "da": "данська",
        "sv": "шведська",
        "no": "норвезька",
        "fi": "фінська",
        "hu": "угорська",
        "ro": "румунська",
        "lt": "литовська",
        "lv": "латвійська",
        "et": "естонська",
        "el": "грецька",
        "he": "іврит",
        "hi": "гінді",
        # Додайте інші мови за потреби
    }


# Функція обробки медіа-файлу - з підтримкою progress_bar
def process_media(media_file_object, language, task, status_object):
    """
    Обробляє завантажений медіа-файл (аудіо або відео), витягує аудіо, якщо потрібно,
    та викликає функцію розшифровки.
    Приймає файловий об'єкт від Streamlit та об'єкт статусу Streamlit.
    """
    if media_file_object is None:
        status_object.update(label="Не завантажено файл", state="error")
        logging.warning("process_media викликано без файлу.")
        return "Файл не завантажено", None, None, None

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
                 return f"Помилка при обробці відео (витягнення аудіо): {e}", None, None, None


        # Розшифрувати аудіо
        status_object.update(label=f"Розшифровка аудіо...", state="running")
        # Можна передати status_object в transcribe_audio, якщо вона підтримує оновлення статусу
        text_output, srt_path, txt_path, segments = transcribe_audio(audio_to_process_path, language, task)

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

        return text_output, srt_path, txt_path, segments

    except Exception as e:
        # Обробка будь-якої іншої помилки під час обробки медіа
        status_object.update(label=f"Виникла неочікувана помилка: {e}", state="error")
        logging.error(f"Загальна помилка в process_media: {e}", exc_info=True)
        return f"Виникла неочікувана помилка: {e}", None, None, None


# Функції для роботи з вихідними файлами
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
        st.error(f"Не вдалося очистити вихідні файли: {e}")
    return get_output_files()


# --- ПОБУДОВА ІНТЕРФЕЙСУ STREAMLIT ---

def main():
    st.title("🎤 Розшифровка та переклад аудіо/відео 🎞️")
    st.markdown("Завантажте аудіо або відео файл для отримання текстової розшифровки, файлу субтитрів (SRT) та перекладу.")
    
    # Використовуємо tabs для розділення функціоналу
    tab1, tab2 = st.tabs(["📝 Розшифровка", "🌐 Переклад"])
    
    with tab1:
        # Розділ розшифровки - використовуємо колонки для більш компактного розміщення
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Завантаження файлу без обмеження типів
            uploaded_file = st.file_uploader(
                "Завантажте аудіо/відео файл",
                accept_multiple_files=False,
                key="transcription_file_uploader"
            )
            
            # Після завантаження файлу - перевіряємо його розширення самостійно
            if uploaded_file is not None:
                filename = uploaded_file.name
                ext = os.path.splitext(filename)[1].lower()
                allowed_extensions = [".wav", ".mp3", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpga", ".mpeg4"]
                
                if ext not in allowed_extensions:
                    st.error(f"Непідтримуваний формат файлу: {ext}. Підтримуються: {', '.join(allowed_extensions)}")
                    uploaded_file = None  # Скидаємо файл, якщо він невідповідного типу
        
        with col2:
            # Вибір моделі Whisper
            whisper_model_size = st.selectbox(
                "Розмір моделі Whisper",
                ["tiny", "base", "small", "medium", "large"],
                index=1,  # "base" за замовчуванням
                help="Більший розмір - краща якість, але більше використання пам'яті і повільніша робота"
            )
            
            # Вибір мови та завдання
            languages = {
                "auto": "Автовизначення",
                "uk": "Українська",
                "en": "Англійська",
                "ru": "Російська", 
                "be": "Білоруська",
                "pl": "Польська",
                "cs": "Чеська",
                "sk": "Словацька",
                "bg": "Болгарська",
                "de": "Німецька", 
                "fr": "Французька", 
                "es": "Іспанська",
                "it": "Італійська",
                "pt": "Португальська",
                "nl": "Нідерландська",
                "da": "Данська",
                "sv": "Шведська",
                "no": "Норвезька",
                "fi": "Фінська",
                "hu": "Угорська",
                "ro": "Румунська",
                "lt": "Литовська",
                "lv": "Латвійська",
                "et": "Естонська",
                "el": "Грецька",
                "tr": "Турецька",
                "ar": "Арабська",
                "he": "Іврит",
                "hi": "Гінді",
                "zh": "Китайська",
                "ja": "Японська",
                "ko": "Корейська"
            }
            
            language = st.selectbox(
                "Мова",
                options=list(languages.keys()),
                format_func=lambda x: languages[x],
                index=list(languages.keys()).index("auto")
            )
            
            task = st.selectbox(
                "Завдання",
                ["transcribe", "translate"],
                format_func=lambda x: "Розшифровка" if x == "transcribe" else "Переклад на англійську",
                index=0
            )
        
        # Кнопка обробки
        process_button = st.button("📝 Обробити файл")
        
        # Використовуємо session_state для збереження результатів між виконаннями скрипта
        if 'transcription_result' not in st.session_state:
            st.session_state.transcription_result = {"text": "", "srt_path": None, "txt_path": None, "segments": None}
        
        # Логіка обробки файлу
        if process_button and uploaded_file is not None:
            # Оновлюємо глобальну модель перед обробкою (якщо розмір моделі змінився)
            global whisper_model
            whisper_model = load_whisper_model(whisper_model_size)
            
            # Використовуємо st.status для відображення прогресу
            with st.status("Початок обробки...", expanded=True) as status:
                # Викликаємо функцію обробки, передаючи файловий об'єкт від Streamlit та об'єкт статусу
                text_output_str, srt_path_result, txt_path_result, segments = process_media(
                    uploaded_file, language, task, status
                )
            
            # Зберігаємо результати в session state
            st.session_state.transcription_result = {
                "text": text_output_str if text_output_str is not None else "",
                "srt_path": srt_path_result,
                "txt_path": txt_path_result,
                "segments": segments
            }
            logging.info("Результати збережено в session_state.")
        
        # Відображення результатів розшифровки
        st.markdown("## 📜 Результат розшифровки:")
        
        # Отримуємо результат з стану
        result_text_from_state = st.session_state.transcription_result.get("text", "")
        if result_text_from_state is None:
            result_text_from_state = ""
        
        # Відображаємо збережений текст
        st.text_area(
            label="Текст",
            value=result_text_from_state,
            height=250,
            disabled=True,
            help="Розшифрований текст"
        )
        
        # Кнопки завантаження файлів
        output_srt_path = st.session_state.transcription_result.get("srt_path")
        output_txt_path = st.session_state.transcription_result.get("txt_path")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if output_srt_path and os.path.exists(output_srt_path):
                try:
                    with open(output_srt_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Завантажити SRT файл",
                            data=f,
                            file_name=os.path.basename(output_srt_path),
                            mime="application/x-subrip",
                            key="download_srt"
                        )
                except Exception as e:
                    st.error(f"Не вдалося підготувати SRT файл для завантаження: {e}")
        
        with col2:
            if output_txt_path and os.path.exists(output_txt_path):
                try:
                    with open(output_txt_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Завантажити TXT файл",
                            data=f,
                            file_name=os.path.basename(output_txt_path),
                            mime="text/plain",
                            key="download_txt"
                        )
                except Exception as e:
                    st.error(f"Не вдалося підготувати TXT файл для завантаження: {e}")
    
    # Вкладка перекладу
    with tab2:
        # Перевіряємо, чи є текст для перекладу в стані сесії
        has_text_to_translate = st.session_state.transcription_result.get("text") not in [None, ""]
        
        if not has_text_to_translate:
            st.info("Спочатку виконайте розшифровку аудіо на вкладці 'Розшифровка', щоб отримати текст для перекладу.")
        else:
            st.markdown("### 🌐 Переклад розшифрованого тексту")
            
            # Отримуємо словник підтримуваних мов
            translation_languages = get_supported_languages()
            
            # Вибір мов для перекладу в двох колонках
            col1, col2 = st.columns(2)
            
            with col1:
                source_lang = st.selectbox(
                    "Мова оригіналу:",
                    options=list(translation_languages.keys()),
                    format_func=lambda x: f"{translation_languages[x]} ({x})",
                    index=list(translation_languages.keys()).index("uk") if "uk" in translation_languages else 0
                )
            
            with col2:
                target_lang = st.selectbox(
                    "Мова перекладу:",
                    options=list(translation_languages.keys()),
                    format_func=lambda x: f"{translation_languages[x]} ({x})",
                    index=list(translation_languages.keys()).index("en") if "en" in translation_languages else 0
                )
            
            # Кнопка перекладу
            translate_button = st.button("🌐 Перекласти текст")
            
            # Відображення тексту для перекладу
            st.markdown("#### Текст для перекладу:")
            st.text_area(
                label="Оригінальний текст",
                value=st.session_state.transcription_result.get("text", ""),
                height=150,
                disabled=True
            )
            
            # Логіка перекладу
            if translate_button:
                with st.status("Виконується переклад...", expanded=True) as status:
                    status.update(label=f"Переклад з {source_lang} на {target_lang}...", state="running")
                    
                    text_to_translate = st.session_state.transcription_result.get("text", "")
                    
                   # Викликаємо функцію перекладу
                    translated_text, translation_path = translate_with_m2m100(text_to_translate, source_lang, target_lang)
                    
                    if isinstance(translated_text, str) and "Помилка" in translated_text:
                        status.update(label=f"Помилка перекладу: {translated_text}", state="error")
                    else:
                        status.update(label="Переклад завершено успішно!", state="complete")
            
                # Відображення результату перекладу
                st.markdown("#### Результат перекладу:")
                
                # Якщо переклад був виконаний, відображаємо результат
                if 'translated_text' in locals() and isinstance(translated_text, str):
                    # Перевіряємо, чи не містить текст повідомлення про помилку
                    if "Помилка" not in translated_text:
                        st.text_area(
                            label="Перекладений текст",
                            value=translated_text,
                            height=150,
                            disabled=True
                        )
                        
                        # Кнопка для завантаження файлу перекладу
                        if translation_path and os.path.exists(translation_path):
                            try:
                                with open(translation_path, "rb") as f:
                                    st.download_button(
                                        label="⬇️ Завантажити переклад (TXT)",
                                        data=f,
                                        file_name=os.path.basename(translation_path),
                                        mime="text/plain",
                                        key="download_translation"
                                    )
                            except Exception as e:
                                st.error(f"Не вдалося підготувати файл перекладу для завантаження: {e}")
                    else:
                        st.error(translated_text)
    
    # Відображення вихідних файлів та інформації про вільне місце
    st.markdown("---")
    st.markdown("## 📁 Управління файлами")
    
    # Получаємо список файлів
    output_files = get_output_files()
    
    # Відображаємо вихідні файли
    if output_files:
        # Створюємо колонки для більш компактного відображення
        st.write(f"Вихідні файли ({len(output_files)}):")
        
        # Використовуємо контейнер з прокруткою для відображення вихідних файлів
        with st.container():
            for file_path in output_files:
                col1, col2, col3 = st.columns([5, 2, 2])
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) / 1024  # розмір в KB
                
                # Формат дати створення файлу
                file_ctime = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
                date_str = file_ctime.strftime("%Y-%m-%d %H:%M:%S")
                
                with col1:
                    st.text(file_name)
                
                with col2:
                    st.text(f"{file_size:.1f} KB")
                
                with col3:
                    try:
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label="⬇️",
                                data=f,
                                file_name=file_name,
                                mime="application/octet-stream",
                                key=f"download_{file_name}"
                            )
                    except Exception as e:
                        st.error(f"Помилка: {e}")
        
        # Кнопка для очищення всіх вихідних файлів
        if st.button("🗑️ Очистити всі вихідні файли"):
            delete_all_output_files()
            st.success("Всі вихідні файли видалено.")
            st.experimental_rerun()
    else:
        st.info("Немає доступних вихідних файлів.")
    
    # Інформація про використання сховища
    st.markdown("### 💾 Інформація про диск")
    
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(OUTPUT_DIR):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        
        # Отримуємо інформацію про диск
        disk = shutil.disk_usage("/")
        
        # Відображаємо інформацію
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Використано (вихідні файли)", f"{total_size / (1024*1024):.2f} MB")
        
        with col2:
            st.metric("Вільно на диску", f"{disk.free / (1024*1024*1024):.1f} GB")
        
        with col3:
            st.metric("Всього на диску", f"{disk.total / (1024*1024*1024):.1f} GB")
    
    except Exception as e:
        st.error(f"Не вдалося отримати інформацію про диск: {e}")
    
    # Додаткова інформація та налаштування
    with st.expander("ℹ️ Інформація про додаток"):
        st.markdown("""
        ### 🎤 Розшифровка та переклад аудіо/відео
        
        Цей додаток дозволяє:
        1. 📝 **Розшифровувати** аудіо та відео файли за допомогою моделі Whisper
        2. 🎬 **Створювати субтитри** у форматі SRT
        3. 🌐 **Перекладати** розшифрований текст на різні мови
        
        **Підтримувані формати файлів:**
        - Аудіо: `.wav`, `.mp3`, `.mpga`
        - Відео: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
        
        **Використовувані технології:**
        - Whisper від OpenAI для розшифровки аудіо/відео
        - M2M-100 від Facebook для перекладу тексту
        
        **Про обмеження:**
        - Більші файли потребують більше часу та пам'яті для обробки
        - Якість розшифровки залежить від якості аудіо та розміру обраної моделі
        - Переклад може мати неточності, особливо для специфічної термінології
        
        **Порада:** Для швидкої обробки використовуйте моделі `tiny` або `base`. Для кращої якості – `small`, `medium` або `large`.
        """)
    
    # Використовуємо два рядки в футері для статистики та авторських прав
    st.markdown("---")
    footer_col1, footer_col2 = st.columns(2)
    
    with footer_col1:
        # Відображаємо поточний час та інформацію про GPU
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gpu_info = "Доступний" if torch.cuda.is_available() else "Недоступний"
        st.write(f"🕒 {current_time} | 🖥️ GPU: {gpu_info}")
    
    with footer_col2:
        # Авторські права та версія
        st.write("© 2023 | Версія 1.0.0 | Зроблено в Україні 🇺🇦")

if __name__ == "__main__":
    main()
