# app.py для Streamlit (Інтегрована версія з розшифровкою та перекладом)

import streamlit as st
import whisper
import os
import shutil # Залишено, хоча прямо не використовується в поточній версії, може знадобитися
import datetime
import time # Залишено, може знадобитися для детального профілювання
import threading # Залишено, може знадобитися для асинхронних задач у майбутньому
import logging
import sys
from moviepy.editor import VideoFileClip # Використовується для обробки відео
import io # Потрібно для читання завантажених файлів
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import tempfile # Залишено, хоча прямо не використовується, TEMP_DIR використовується для тимчасових файлів

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
    st.error("Критична помилка: не вдалося завантажити основну модель Whisper. Додаток не може продовжити роботу.")
    st.stop()


# --- ФУНКЦІЇ ОБРОБКИ ---

def transcribe_audio(audio_path, language=None, task="transcribe"):
    """
    Функція для розшифровки аудіо файлу за допомогою Whisper.
    Використовує глобальну змінну 'whisper_model'.
    """
    logging.info(f"Запуск розшифровки/перекладу для: {os.path.basename(audio_path)}, мова: {language}, завдання: {task}")
    try:
        global whisper_model
        if whisper_model is None:
            # Спроба повторно завантажити модель, якщо вона чомусь None
            # Це може статися, якщо розмір моделі змінили і перше завантаження не вдалося
            # або якщо st.cache_resource очистився
            active_model_size = st.session_state.get('current_whisper_model_size', 'base') # Отримати розмір, якщо є
            whisper_model = load_whisper_model(active_model_size)
            if whisper_model is None:
                 raise RuntimeError(f"Модель Whisper ({active_model_size}) не завантажена і не може бути завантажена повторно.")


        result = whisper_model.transcribe(audio_path, language=language if language != "auto" else None, task=task)
        text = result["text"]

        base_filename = os.path.basename(audio_path).rsplit('.', 1)[0]
        txt_path = os.path.join(OUTPUT_DIR, base_filename + ".txt")
        srt_path = os.path.join(OUTPUT_DIR, base_filename + ".srt")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"Текст збережено до: {txt_path}")

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"]):
                start_td = datetime.timedelta(seconds=segment['start'])
                end_td = datetime.timedelta(seconds=segment['end'])
                
                # Обробка для формату HH:MM:SS.mmm
                start_total_seconds = segment['start']
                start_hours = int(start_total_seconds // 3600)
                start_minutes = int((start_total_seconds % 3600) // 60)
                start_seconds = int(start_total_seconds % 60)
                start_milliseconds = int((start_total_seconds - int(start_total_seconds)) * 1000)

                end_total_seconds = segment['end']
                end_hours = int(end_total_seconds // 3600)
                end_minutes = int((end_total_seconds % 3600) // 60)
                end_seconds = int(end_total_seconds % 60)
                end_milliseconds = int((end_total_seconds - int(end_total_seconds)) * 1000)

                start_str = f"{start_hours:02}:{start_minutes:02}:{start_seconds:02},{start_milliseconds:03}"
                end_str = f"{end_hours:02}:{end_minutes:02}:{end_seconds:02},{end_milliseconds:03}"
                
                f.write(f"{i+1}\n{start_str} --> {end_str}\n{segment['text'].strip()}\n\n")
        logging.info(f"Субтитри збережено до: {srt_path}")

        logging.info("Розшифровка/переклад завершено успішно.")
        return text, srt_path, txt_path, result["segments"]

    except Exception as e:
        logging.error(f"Помилка під час розшифровки/перекладу transcribe_audio: {e}", exc_info=True)
        return f"Помилка під час розшифровки: {e}", None, None, None


# Функція для перекладу тексту за допомогою M2M-100
def translate_with_m2m100(text, source_lang, target_lang):
    """
    Перекладає текст з вихідної мови на цільову за допомогою моделі M2M-100.
    """
    logging.info(f"Запуск перекладу тексту з {source_lang} на {target_lang}")
    # Розмір моделі M2M100 фіксований у цьому випадку, але можна зробити його вибором користувача
    model, tokenizer = load_m2m100_model("facebook/m2m100_418M")
    
    if not model or not tokenizer:
        return "Не вдалося завантажити модель перекладу", None
    
    try:
        tokenizer.src_lang = source_lang
        
        encoded_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024) # Збільшено max_length
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded_text,
                forced_bos_token_id=tokenizer.get_lang_id(target_lang),
                max_length=1024 # Збільшено max_length
            )
        
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
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
    # Це приблизний список, повний список залежить від моделі M2M-100
    return {
        "uk": "українська", "en": "англійська", "ru": "російська", "de": "німецька",
        "fr": "французька", "es": "іспанська", "pl": "польська", "it": "італійська",
        "cs": "чеська", "ja": "японська", "zh": "китайська", "ko": "корейська",
        "ar": "арабська", "tr": "турецька", "vi": "в'єтнамська", "pt": "португальська",
        "be": "білоруська", "sk": "словацька", "bg": "болгарська", "nl": "нідерландська",
        "da": "данська", "sv": "шведська", "no": "норвезька", "fi": "фінська",
        "hu": "угорська", "ro": "румунська", "lt": "литовська", "lv": "латвійська",
        "et": "естонська", "el": "грецька", "he": "іврит", "hi": "гінді",
        # Додайте інші мови, якщо модель їх підтримує і вони потрібні
    }


# Функція обробки медіа-файлу - з підтримкою progress_bar
def process_media(media_file_object, language, task, status_object):
    if media_file_object is None:
        status_object.update(label="Не завантажено файл", state="error")
        logging.warning("process_media викликано без файлу.")
        return "Файл не завантажено", None, None, None

    logging.info(f"Отримано файл для обробки: {media_file_object.name}")
    status_object.update(label=f"Отримано файл '{media_file_object.name}'", state="running", expanded=True)

    filename = media_file_object.name
    temp_input_path = os.path.join(TEMP_DIR, filename)

    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        status_object.update(label=f"Збереження файлу '{filename}'...", state="running")
        with open(temp_input_path, "wb") as f:
            f.write(media_file_object.getbuffer())
        logging.info(f"Файл тимчасово збережено до: {temp_input_path}")
        status_object.write(f"Файл тимчасово збережено до: {os.path.basename(temp_input_path)}")

        ext = os.path.splitext(filename)[1].lower()
        audio_to_process_path = temp_input_path

        # Відео формати, з яких потрібно витягувати аудіо (на основі ваших allowed_extensions)
        video_formats_for_extraction = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg4"]

        if ext in video_formats_for_extraction:
            try:
                status_object.update(label=f"Витягування аудіо з відео ({ext})...", state="running")
                logging.info(f"Виявлено відеоформат ({ext}), витягування аудіо...")
                video = VideoFileClip(temp_input_path)
                audio_filename = os.path.splitext(filename)[0] + ".wav" # Завжди .wav для сумісності
                audio_to_process_path = os.path.join(TEMP_DIR, audio_filename)
                logging.info(f"Витягування аудіо до: {audio_to_process_path}")
                status_object.write(f"Витягування аудіо до: {os.path.basename(audio_to_process_path)}")
                video.audio.write_audiofile(audio_to_process_path, codec='pcm_s16le', verbose=False, logger=None)
                video.close()
                logging.info("Аудіо витягнуто успішно.")
                status_object.write("Аудіо витягнуто успішно.")
            except Exception as e:
                status_object.update(label="Помилка при витягуванні аудіо", state="error")
                logging.error(f"Помилка при витягуванні аудіо з відео {filename}: {e}", exc_info=True)
                # Не видаляємо temp_input_path тут, щоб уникнути помилок при очищенні в finally
                return f"Помилка при обробці відео (витягнення аудіо): {e}", None, None, None
        elif ext in [".wav", ".mp3", ".mpga"]: # Явні аудіо формати
             logging.info(f"Файл '{filename}' є аудіоформатом ({ext}). Витягування не потрібне.")
             status_object.write(f"Файл є аудіоформатом ({ext}). Обробка напряму.")
        else:
            logging.warning(f"Файл '{filename}' має розширення ({ext}), яке не є явно відео або аудіо. Спроба обробки як є.")
            status_object.write(f"Формат файлу ({ext}). Спроба прямої обробки.")


        status_object.update(label=f"Розшифровка аудіо з '{os.path.basename(audio_to_process_path)}'...", state="running")
        text_output, srt_path, txt_path, segments = transcribe_audio(audio_to_process_path, language, task)

        if srt_path is None and txt_path is None: # Ознака помилки з transcribe_audio
            status_object.update(label=f"Помилка під час розшифровки: {text_output}", state="error")
            # text_output вже містить повідомлення про помилку
        else:
            status_object.update(label="Обробка завершена!", state="complete")
        
        return text_output, srt_path, txt_path, segments

    except Exception as e:
        status_object.update(label=f"Виникла неочікувана помилка: {e}", state="error")
        logging.error(f"Загальна помилка в process_media: {e}", exc_info=True)
        return f"Виникла неочікувана помилка: {e}", None, None, None
    finally:
        # Очищення тимчасових файлів
        status_object.write("Очищення тимчасових файлів...")
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
                logging.info(f"Видалено тимчасовий вхідний файл: {temp_input_path}")
                status_object.write(f"Видалено тимчасовий вхідний файл: {os.path.basename(temp_input_path)}")
            except Exception as e_clean:
                 logging.error(f"Не вдалося видалити тимчасовий файл {temp_input_path}: {e_clean}")

        # Видаляємо витягнутий аудіофайл, якщо він відрізняється від вхідного
        if audio_to_process_path != temp_input_path and os.path.exists(audio_to_process_path):
            try:
                os.remove(audio_to_process_path)
                logging.info(f"Видалено тимчасовий аудіо файл: {audio_to_process_path}")
                status_object.write(f"Видалено тимчасовий аудіо файл: {os.path.basename(audio_to_process_path)}")
            except Exception as e_clean_audio:
                logging.error(f"Не вдалося видалити тимчасовий аудіофайл {audio_to_process_path}: {e_clean_audio}")


# Функції для роботи з вихідними файлами
def get_output_files():
    logging.info(f"Отримання списку файлів з {OUTPUT_DIR}")
    if not os.path.exists(OUTPUT_DIR):
        return []
    return [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))]

def delete_all_output_files():
    logging.info(f"Видалення всіх файлів у {OUTPUT_DIR}")
    deleted_count = 0
    errors_count = 0
    try:
        if os.path.exists(OUTPUT_DIR):
            for f_name in os.listdir(OUTPUT_DIR):
                f_path = os.path.join(OUTPUT_DIR, f_name)
                if os.path.isfile(f_path):
                    try:
                        os.remove(f_path)
                        deleted_count +=1
                    except Exception as e_del:
                        logging.error(f"Не вдалося видалити файл {f_path}: {e_del}")
                        errors_count +=1
            if errors_count == 0:
                logging.info(f"Усі {deleted_count} вихідні файли видалено успішно.")
            else:
                logging.warning(f"{deleted_count} файлів видалено, {errors_count} помилок при видаленні.")
        else:
            logging.info("Вихідна директорія не існує, видалення не потрібне.")
    except Exception as e:
        logging.error(f"Помилка при видаленні вихідних файлів: {e}", exc_info=True)
        st.error(f"Не вдалося очистити вихідні файли: {e}")
    return get_output_files()


# --- ПОБУДОВА ІНТЕРФЕЙСУ STREAMLIT ---

def main():
    st.set_page_config(page_title="Розшифровка та Переклад", layout="wide")
    st.title("🎤 Розшифровка та переклад аудіо/відео 🎞️")
    st.markdown("Завантажте аудіо або відео файл для отримання текстової розшифровки, файлу субтитрів (SRT) та перекладу.")
    
    # Ініціалізація станів сесії, якщо вони ще не існують
    if 'transcription_result' not in st.session_state:
        st.session_state.transcription_result = {"text": "", "srt_path": None, "txt_path": None, "segments": None}
    if 'translation_result' not in st.session_state:
        st.session_state.translation_result = {"text": "", "path": None}
    if 'current_whisper_model_size' not in st.session_state:
        st.session_state.current_whisper_model_size = "base"


    tab1, tab2 = st.tabs(["📝 Розшифровка", "🌐 Переклад"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Завантажте аудіо/відео файл",
                accept_multiple_files=False,
                key="transcription_file_uploader"
            )
            
            valid_file_uploaded = False
            if uploaded_file is not None:
                filename = uploaded_file.name
                ext = os.path.splitext(filename)[1].lower()
                # .mpeg4 також додано для узгодження з process_media
                allowed_extensions = [".wav", ".mp3", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpga", ".mpeg4", ".mpeg"]
                
                if ext not in allowed_extensions:
                    st.error(f"Непідтримуваний формат файлу: {ext}. Підтримуються: {', '.join(allowed_extensions)}")
                else:
                    valid_file_uploaded = True
        
        with col2:
            whisper_model_options = ["tiny", "base", "small", "medium", "large"]
            default_whisper_idx = whisper_model_options.index(st.session_state.current_whisper_model_size)

            whisper_model_size_choice = st.selectbox(
                "Розмір моделі Whisper",
                whisper_model_options,
                index=default_whisper_idx,
                help="Більший розмір - краща якість, але більше використання пам'яті і повільніша робота"
            )
            
            languages = {
                "auto": "Автовизначення", "uk": "Українська", "en": "Англійська", "ru": "Російська",
                "be": "Білоруська", "pl": "Польська", "cs": "Чеська", "sk": "Словацька",
                "bg": "Болгарська", "de": "Німецька", "fr": "Французька", "es": "Іспанська",
                "it": "Італійська", "pt": "Португальська", "nl": "Нідерландська", "da": "Данська",
                "sv": "Шведська", "no": "Норвезька", "fi": "Фінська", "hu": "Угорська",
                "ro": "Румунська", "lt": "Литовська", "lv": "Латвійська", "et": "Естонська",
                "el": "Грецька", "tr": "Турецька", "ar": "Арабська", "he": "Іврит",
                "hi": "Гінді", "zh": "Китайська", "ja": "Японська", "ko": "Корейська"
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
                format_func=lambda x: "Розшифровка" if x == "transcribe" else "Переклад на англійську (Whisper)",
                index=0
            )
        
        process_button = st.button("📝 Обробити файл", disabled=not valid_file_uploaded)
        
        if process_button and valid_file_uploaded:
            if whisper_model_size_choice != st.session_state.current_whisper_model_size:
                with st.spinner(f"Завантаження моделі Whisper: {whisper_model_size_choice}... Це може зайняти деякий час."):
                    global whisper_model # Оголошуємо, що будемо змінювати глобальну змінну
                    whisper_model = load_whisper_model(whisper_model_size_choice) # Завантажуємо нову модель
                    st.session_state.current_whisper_model_size = whisper_model_size_choice
                if whisper_model is None:
                    st.error(f"Не вдалося завантажити обрану модель Whisper ({whisper_model_size_choice}). Спробуйте іншу або перезапустіть.")
                    st.stop()

            with st.status("Початок обробки...", expanded=True) as status:
                text_output_str, srt_path_result, txt_path_result, segments_result = process_media(
                    uploaded_file, language, task, status
                )
            
            st.session_state.transcription_result = {
                "text": text_output_str if text_output_str is not None else "",
                "srt_path": srt_path_result,
                "txt_path": txt_path_result,
                "segments": segments_result
            }
            logging.info("Результати розшифровки збережено в session_state.")
        
        st.markdown("## 📜 Результат розшифровки:")
        
        result_text_from_state = st.session_state.transcription_result.get("text", "")
        result_srt_path = st.session_state.transcription_result.get("srt_path")
        result_txt_path = st.session_state.transcription_result.get("txt_path")

        if result_srt_path is None and result_txt_path is None and result_text_from_state.startswith("Помилка"):
            st.error(result_text_from_state) # Відображаємо помилку, якщо вона сталася
            st.text_area(label="Текст", value="", height=250, disabled=True)
        else:
            st.text_area(
                label="Текст",
                value=result_text_from_state,
                height=250,
                disabled=True, # Зробити нередагованим
                help="Розшифрований текст"
            )
        
        dl_col1, dl_col2 = st.columns(2)
        
        with dl_col1:
            if result_srt_path and os.path.exists(result_srt_path):
                try:
                    with open(result_srt_path, "rb") as f_srt:
                        st.download_button(
                            label="⬇️ Завантажити SRT файл",
                            data=f_srt,
                            file_name=os.path.basename(result_srt_path),
                            mime="application/x-subrip",
                            key="download_srt"
                        )
                except Exception as e:
                    st.error(f"Не вдалося підготувати SRT файл для завантаження: {e}")
        
        with dl_col2:
            if result_txt_path and os.path.exists(result_txt_path):
                try:
                    with open(result_txt_path, "rb") as f_txt:
                        st.download_button(
                            label="⬇️ Завантажити TXT файл",
                            data=f_txt,
                            file_name=os.path.basename(result_txt_path),
                            mime="text/plain",
                            key="download_txt"
                        )
                except Exception as e:
                    st.error(f"Не вдалося підготувати TXT файл для завантаження: {e}")

    with tab2:
        has_text_to_translate = st.session_state.transcription_result.get("text") not in [None, ""] and \
                                not st.session_state.transcription_result.get("text", "").startswith("Помилка")
        
        if not has_text_to_translate:
            st.info("Спочатку виконайте успішну розшифровку аудіо на вкладці 'Розшифровка', щоб отримати текст для перекладу.")
        else:
            st.markdown("### 🌐 Переклад розшифрованого тексту (M2M-100)")
            
            translation_languages = get_supported_languages()
            
            col_lang1, col_lang2 = st.columns(2)
            
            with col_lang1:
                # Визначення мови оригіналу (можна спробувати визначити автоматично з Whisper)
                detected_lang_whisper = None
                if st.session_state.transcription_result.get("segments"):
                    try:
                        # Whisper's result has 'language' field
                        detected_lang_whisper = whisper_model.detect_language(whisper.pad_or_trim(whisper.load_audio(st.session_state.transcription_result.get("txt_path").replace(".txt", ".wav")))).language # Припускаємо, що є .wav
                    except Exception: # Якщо вихідний файл не .wav або інша помилка
                        pass

                default_source_lang_idx = 0
                if detected_lang_whisper and detected_lang_whisper in translation_languages:
                     default_source_lang_idx = list(translation_languages.keys()).index(detected_lang_whisper)
                elif "uk" in translation_languages: # Якщо не вдалося, ставимо українську
                    default_source_lang_idx = list(translation_languages.keys()).index("uk")


                source_lang = st.selectbox(
                    "Мова оригіналу:",
                    options=list(translation_languages.keys()),
                    format_func=lambda x: f"{translation_languages[x]} ({x})",
                    index=default_source_lang_idx, # Українська за замовчуванням або визначена
                    key="source_language_select"
                )
            
            with col_lang2:
                default_target_lang_idx = list(translation_languages.keys()).index("en") if "en" in translation_languages else 0
                target_lang = st.selectbox(
                    "Мова перекладу:",
                    options=list(translation_languages.keys()),
                    format_func=lambda x: f"{translation_languages[x]} ({x})",
                    index=default_target_lang_idx, # Англійська за замовчуванням
                    key="target_language_select"
                )
            
            st.markdown("#### Текст для перекладу:")
            original_text_for_translation = st.session_state.transcription_result.get("text", "")
            st.text_area(
                label="Оригінальний текст",
                value=original_text_for_translation,
                height=150,
                disabled=True,
                key="original_text_for_translation_display"
            )

            translate_button = st.button("🌐 Перекласти текст", key="translate_m2m100_button")
            
            if translate_button:
                if not original_text_for_translation:
                    st.error("Немає тексту для перекладу.")
                elif source_lang == target_lang:
                    st.warning("Мова оригіналу та мова перекладу однакові. Переклад не потрібен.")
                    st.session_state.translation_result = {
                        "text": original_text_for_translation,
                        "path": None 
                    }
                else:
                    with st.status(f"Переклад з '{translation_languages[source_lang]}' на '{translation_languages[target_lang]}'...", expanded=True) as status_translate:
                        translated_text_result, translation_path_result = translate_with_m2m100(
                            original_text_for_translation, source_lang, target_lang
                        )
                        
                        if translation_path_result is not None: # Успіх
                            st.session_state.translation_result = {
                                "text": translated_text_result,
                                "path": translation_path_result
                            }
                            status_translate.update(label="Переклад завершено!", state="complete")
                            logging.info(f"Переклад успішний: {translation_path_result}")
                        else: # Помилка під час перекладу
                            st.session_state.translation_result = {
                                "text": translated_text_result, # Містить повідомлення про помилку
                                "path": None
                            }
                            status_translate.update(label=f"Помилка перекладу: {translated_text_result}", state="error")
                            logging.error(f"Помилка перекладу M2M: {translated_text_result}")
            
            st.markdown("#### Результат перекладу:")
            current_translated_text = st.session_state.translation_result.get("text", "")
            current_translation_path = st.session_state.translation_result.get("path")

            if current_translation_path is None and \
               (isinstance(current_translated_text, str) and (current_translated_text.startswith("Помилка") or current_translated_text.startswith("Не вдалося завантажити"))):
                st.error(current_translated_text) # Відображаємо помилку
                st.text_area(label="Перекладений текст", value="", height=150, disabled=True, key="translated_text_display_error")
            else:
                st.text_area(
                    label="Перекладений текст",
                    value=current_translated_text,
                    height=150,
                    disabled=True, # Зробити нередагованим
                    key="translated_text_display_success"
                )

            if current_translation_path and os.path.exists(current_translation_path):
                try:
                    with open(current_translation_path, "rb") as f_trans:
                        st.download_button(
                            label="⬇️ Завантажити переклад (TXT)",
                            data=f_trans,
                            file_name=os.path.basename(current_translation_path),
                            mime="text/plain",
                            key="download_translation_txt_button"
                        )
                except Exception as e:
                    st.error(f"Не вдалося підготувати файл перекладу для завантаження: {e}")
            elif source_lang == target_lang and current_translated_text: # Для випадку, коли мови однакові
                 pass # Кнопка завантаження не потрібна, бо файл не створювався


    # --- Розділ керування вихідними файлами на бічній панелі ---
    st.sidebar.title("🗂️ Керування файлами")
    if st.sidebar.button("🗑️ Видалити всі вихідні файли"):
        delete_all_output_files()
        st.sidebar.success("Усі вихідні файли видалено!")
        # Очистити стан результатів, щоб кнопки завантаження зникли
        st.session_state.transcription_result = {"text": "", "srt_path": None, "txt_path": None, "segments": None}
        st.session_state.translation_result = {"text": "", "path": None}
        st.rerun() # Перезапустити додаток, щоб оновити UI

    st.sidebar.markdown("### 📄 Збережені файли:")
    output_files_list = get_output_files()
    if output_files_list:
        for f_path_item in output_files_list:
            st.sidebar.markdown(f"- `{os.path.basename(f_path_item)}`")
    else:
        st.sidebar.info("Немає збережених файлів.")

    # --- Додавання нижнього колонтитулу ---
    st.markdown("---")
    st.markdown("<div style='text-align: center;'>Розроблено з ❤️ за допомогою Streamlit, Whisper та Transformers.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
