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
        # Додаємо атрибут name до моделі
        model.name = model_name
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

# Завантажуємо модель Whisper при старті додатку
whisper_model_instance = load_whisper_model("base") # Змінено назву змінної, щоб уникнути конфлікту з глобальною

# Перевіряємо, чи модель завантажилась успішно, перед тим як продовжувати
if whisper_model_instance is None:
    st.stop()


# --- ФУНКЦІЇ ОБРОБКИ ---

def transcribe_audio(audio_path, whisper_model_to_use, language=None, task="transcribe"):
    """
    Функція для розшифровки аудіо файлу за допомогою Whisper.
    """
    logging.info(f"Запуск розшифровки/перекладу для: {os.path.basename(audio_path)}, мова: {language}, завдання: {task}")
    try:
        if whisper_model_to_use is None:
            raise RuntimeError("Модель Whisper не завантажена.")

        result = whisper_model_to_use.transcribe(audio_path, language=language if language != "auto" else None, task=task)
        text = result["text"]
        detected_language = result.get("language") # Отримуємо визначену мову

        # Створення шляху для вихідного TXT файлу
        base_filename = os.path.basename(audio_path).rsplit('.', 1)[0] # Отримуємо ім'я без розширення
        txt_path = os.path.join(OUTPUT_DIR, base_filename + ".txt")

        # Переконайтеся, що OUTPUT_DIR існує перед записом
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Збереження тексту в TXT
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"Текст збережено до: {txt_path}")

        logging.info("Розшифровка/переклад Whisper завершено успішно.")
        return text, txt_path, detected_language, result["segments"]

    except Exception as e:
        logging.error(f"Помилка під час розшифровки/перекладу transcribe_audio: {e}", exc_info=True)
        return f"Помилка під час розшифровки Whisper: {e}", None, None, None


# Функція для перекладу тексту за допомогою M2M-100
def translate_with_m2m100(text, source_lang, target_lang):
    """
    Перекладає текст з вихідної мови на цільову за допомогою моделі M2M-100.
    """
    if not text or not source_lang or not target_lang:
        logging.warning("Функцію перекладу викликано з неповними даними.")
        return "Текст, вихідна або цільова мова не вказані.", None

    logging.info(f"Запуск перекладу тексту з '{source_lang}' на '{target_lang}'")
    model, tokenizer = load_m2m100_model() 
    
    if not model or not tokenizer:
        return "Не вдалося завантажити модель перекладу M2M-100", None
    
    try:
        # Встановлюємо вихідну мову для токенізатора
        # M2M100 очікує коди мов зі списку tokenizer.langs
        if source_lang not in tokenizer.langs:
            logging.error(f"Мова оригіналу '{source_lang}' не підтримується токенізатором M2M100.")
            # Спробуємо знайти схожий код, наприклад, 'uk_UA' -> 'uk'
            simple_source_lang = source_lang.split('_')[0]
            if simple_source_lang in tokenizer.langs:
                logging.info(f"Використовується спрощений код мови оригіналу: '{simple_source_lang}'")
                tokenizer.src_lang = simple_source_lang
            else:
                return f"Мова оригіналу '{source_lang}' не підтримується M2M100.", None
        else:
            tokenizer.src_lang = source_lang
        
        # Токенізуємо вхідний текст
        encoded_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024) # Збільшено max_length
        
        # Генеруємо переклад
        with torch.no_grad():
            # Встановлюємо токен цільової мови для генерації
            if target_lang not in tokenizer.langs:
                logging.error(f"Цільова мова '{target_lang}' не підтримується токенізатором M2M100.")
                simple_target_lang = target_lang.split('_')[0]
                if simple_target_lang in tokenizer.langs:
                     logging.info(f"Використовується спрощений код цільової мови: '{simple_target_lang}'")
                     target_lang_id = tokenizer.get_lang_id(simple_target_lang)
                else:
                    return f"Цільова мова '{target_lang}' не підтримується M2M100.", None
            else:
                target_lang_id = tokenizer.get_lang_id(target_lang)

            generated_tokens = model.generate(
                **encoded_text,
                forced_bos_token_id=target_lang_id,
                max_length=1024 # Збільшено max_length
            )
        
        # Декодуємо результат
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        # Збереження перекладу в TXT файл
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        translation_filename = f"m2m_translation_{source_lang}_to_{target_lang}_{timestamp}.txt"
        translation_path = os.path.join(OUTPUT_DIR, translation_filename)
        
        with open(translation_path, "w", encoding="utf-8") as f:
            f.write(translated_text)
        
        logging.info(f"Переклад M2M100 успішно збережено до: {translation_path}")
        return translated_text, translation_path
    
    except Exception as e:
        error_msg = f"Помилка під час перекладу M2M100: {e}"
        logging.error(error_msg, exc_info=True)
        return error_msg, None


# Функція для отримання підтримуваних мов M2M-100
def get_supported_languages_m2m100():
    """Повертає словник підтримуваних мов M2M-100 для UI."""
    # Це базовий список, M2M100 підтримує більше, але ці коди мають працювати
    return {
        "uk": "українська", "en": "англійська", "pl": "польська", "de": "німецька",
        "fr": "французька", "es": "іспанська", "it": "італійська", "ru": "російська",
        "cs": "чеська", "ja": "японська", "zh": "китайська", "ko": "корейська",
        "ar": "арабська", "tr": "турецька", "vi": "в'єтнамська", "pt": "португальська",
        "be": "білоруська", "sk": "словацька", "bg": "болгарська", "nl": "нідерландська", 
        "da": "данська", "sv": "шведська", "no": "норвезька", "fi": "фінська",
        "hu": "угорська", "ro": "румунська", "lt": "литовська", "lv": "латвійська",
        "et": "естонська", "el": "грецька", "he": "іврит", "hi": "гінді",
    }

# Функція обробки медіа-файлу
def process_media(media_file_object, whisper_model_to_use, whisper_language_option, whisper_task_option, target_m2m_lang_option, status_object):
    """
    Обробляє завантажений медіа-файл, розшифровує та опціонально перекладає.
    """
    if media_file_object is None:
        status_object.update(label="Не завантажено файл", state="error")
        logging.warning("process_media викликано без файлу.")
        return "Файл не завантажено", None, None, None, None, None

    logging.info(f"Отримано файл для обробки: {media_file_object.name}")
    status_object.update(label=f"Отримано файл '{media_file_object.name}'", state="running", expanded=True)

    filename = media_file_object.name
    temp_input_path = os.path.join(TEMP_DIR, f"input_{filename}") # Додано префікс, щоб уникнути конфліктів

    # Ініціалізація змінних результатів
    text_output_whisper = None
    txt_path_whisper = None
    detected_lang_whisper = None
    segments_whisper = None
    translated_text_m2m = None
    translation_m2m_path = None

    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        status_object.update(label=f"Збереження файлу '{filename}'...", state="running")
        with open(temp_input_path, "wb") as f:
            f.write(media_file_object.getbuffer())
        logging.info(f"Файл тимчасово збережено до: {temp_input_path}")
        status_object.write(f"Файл тимчасово збережено до: {os.path.basename(temp_input_path)}")

        ext = os.path.splitext(filename)[1].lower()
        audio_to_process_path = temp_input_path

        if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpga", ".mpeg", ".mpeg4"]:
            try:
                status_object.update(label=f"Витягування аудіо з відео ({ext})...", state="running")
                video = VideoFileClip(temp_input_path)
                audio_filename = os.path.splitext(filename)[0] + ".wav"
                # Зберігаємо витягнуте аудіо у TEMP_DIR з унікальним іменем
                audio_to_process_path = os.path.join(TEMP_DIR, f"audio_ext_{audio_filename}")
                logging.info(f"Витягування аудіо до: {audio_to_process_path}")
                status_object.write(f"Витягування аудіо до: {os.path.basename(audio_to_process_path)}")
                video.audio.write_audiofile(audio_to_process_path, codec='pcm_s16le', verbose=False, logger=None)
                video.close() # Закриваємо відеофайл
                logging.info("Аудіо витягнуто успішно.")
                status_object.write("Аудіо витягнуто успішно.")
            except Exception as e:
                status_object.update(label="Помилка при витягуванні аудіо", state="error")
                logging.error(f"Помилка при витягуванні аудіо з відео {filename}: {e}", exc_info=True)
                return f"Помилка при обробці відео (витягнення аудіо): {e}", None, None, None, None, None
        
        status_object.update(label=f"Розшифровка аудіо ({whisper_task_option})...", state="running")
        text_output_whisper, txt_path_whisper, detected_lang_whisper, segments_whisper = transcribe_audio(
            audio_to_process_path, whisper_model_to_use, whisper_language_option, whisper_task_option
        )

        if txt_path_whisper is None or "Помилка під час розшифровки Whisper" in text_output_whisper:
            status_object.update(label=f"Помилка Whisper: {text_output_whisper}", state="error")
            # Не продовжуємо, якщо розшифровка не вдалася
        else:
            status_object.write(f"Розшифровка Whisper завершена. Визначена мова: {detected_lang_whisper if detected_lang_whisper else 'не визначено'}")
            
            # Автоматичний переклад M2M100, якщо обрано
            if target_m2m_lang_option != "none" and text_output_whisper:
                status_object.update(label=f"Запуск автоматичного перекладу M2M100 на '{target_m2m_lang_option}'...", state="running")
                
                source_lang_for_m2m100 = None
                if whisper_task_option == "translate": # Whisper переклав на англійську
                    source_lang_for_m2m100 = "en"
                elif whisper_language_option == "auto":
                    source_lang_for_m2m100 = detected_lang_whisper
                else:
                    source_lang_for_m2m100 = whisper_language_option
                
                if source_lang_for_m2m100:
                    status_object.write(f"M2M100: Переклад з '{source_lang_for_m2m100}' на '{target_m2m_lang_option}'.")
                    translated_text_m2m, translation_m2m_path = translate_with_m2m100(
                        text_output_whisper, source_lang_for_m2m100, target_m2m_lang_option
                    )
                    if translation_m2m_path:
                        status_object.write(f"Автоматичний переклад M2M100 завершено: {translation_m2m_path}")
                    else:
                        status_object.write(f"Помилка автоматичного перекладу M2M100: {translated_text_m2m}")
                else:
                    status_object.write("Не вдалося визначити вихідну мову для автоматичного перекладу M2M100.")
                    translated_text_m2m = "Не вдалося визначити вихідну мову для перекладу."
            
            status_object.update(label="Обробка завершена!", state="complete")

    except Exception as e:
        status_object.update(label=f"Виникла неочікувана помилка: {e}", state="error")
        logging.error(f"Загальна помилка в process_media: {e}", exc_info=True)
        # Повертаємо поточні значення, навіть якщо вони None
        return f"Виникла неочікувана помилка: {e}", txt_path_whisper, detected_lang_whisper, segments_whisper, translated_text_m2m, translation_m2m_path
    finally:
        # Очищення тимчасових файлів
        status_object.write("Очищення тимчасових файлів...")
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
                logging.info(f"Видалено тимчасовий вхідний файл: {temp_input_path}")
            except Exception as e_clean:
                 logging.error(f"Не вдалося видалити {temp_input_path}: {e_clean}")

        if audio_to_process_path != temp_input_path and os.path.exists(audio_to_process_path):
            try:
                os.remove(audio_to_process_path)
                logging.info(f"Видалено тимчасовий аудіо файл: {audio_to_process_path}")
            except Exception as e_clean_audio:
                logging.error(f"Не вдалося видалити {audio_to_process_path}: {e_clean_audio}")


    return text_output_whisper, txt_path_whisper, detected_lang_whisper, segments_whisper, translated_text_m2m, translation_m2m_path


# Функції для роботи з вихідними файлами
def get_output_files():
    logging.info(f"Отримання списку файлів з {OUTPUT_DIR}")
    if not os.path.exists(OUTPUT_DIR):
        return []
    return [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))]

def delete_all_output_files():
    logging.info(f"Видалення всіх файлів у {OUTPUT_DIR}")
    try:
        if os.path.exists(OUTPUT_DIR):
            for f_name in os.listdir(OUTPUT_DIR):
                f_path = os.path.join(OUTPUT_DIR, f_name)
                if os.path.isfile(f_path):
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
    st.set_page_config(layout="wide")
    st.title("🎤 Розшифровка та переклад аудіо/відео 🎞️")
    st.markdown("Завантажте аудіо або відео файл для отримання текстової розшифровки та автоматичного перекладу.")
    
    # Ініціалізація стану сесії
    if 'transcription_processing_result' not in st.session_state:
        st.session_state.transcription_processing_result = {
            "text_whisper": "", 
            "txt_path_whisper": None, 
            "detected_language_whisper": None, 
            "segments_whisper": None,
            "auto_translated_text_m2m": None,
            "auto_translation_m2m_path": None
        }

    tab1, tab2 = st.tabs(["📝 Розшифровка та Авто-переклад", "🌐 Ручний Переклад"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Завантажте аудіо/відео файл",
                accept_multiple_files=False,
                key="transcription_file_uploader"
            )
            
            if uploaded_file is not None:
                filename = uploaded_file.name
                ext = os.path.splitext(filename)[1].lower()
                allowed_extensions = [".wav", ".mp3", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpga", ".mpeg", ".mpeg4"]
                if ext not in allowed_extensions:
                    st.error(f"Непідтримуваний формат файлу: {ext}. Підтримуються: {', '.join(allowed_extensions)}")
                    uploaded_file = None 
        
        with col2:
            whisper_model_size_option = st.selectbox(
                "Розмір моделі Whisper",
                ["tiny", "base", "small", "medium", "large"],
                index=1,
                help="Більший розмір - краща якість, але більше використання пам'яті і повільніша робота"
            )
            
            whisper_languages = {
                "auto": "Автовизначення", "uk": "Українська", "en": "Англійська", "ru": "Російська", 
                "pl": "Польська", "de": "Німецька", # Додайте більше за потреби
            }
            whisper_language_option = st.selectbox(
                "Мова для Whisper",
                options=list(whisper_languages.keys()),
                format_func=lambda x: whisper_languages[x],
                index=0
            )
            
            whisper_task_option = st.selectbox(
                "Завдання для Whisper",
                ["transcribe", "translate"],
                format_func=lambda x: "Розшифровка" if x == "transcribe" else "Переклад на англійську (Whisper)",
                index=0
            )

            m2m100_target_languages = {"none": "Не перекладати"}
            m2m100_target_languages.update(get_supported_languages_m2m100())
            
            target_m2m_lang_choice = st.selectbox(
                "Цільова мова для автоматичного перекладу (M2M100)",
                options=list(m2m100_target_languages.keys()),
                format_func=lambda x: m2m100_target_languages[x],
                index=0 # "Не перекладати" за замовчуванням
            )

        process_button = st.button("🚀 Обробити файл", use_container_width=True)
        
        if process_button and uploaded_file is not None:
            global whisper_model_instance # Використовуємо глобальну змінну для моделі Whisper
            if whisper_model_instance.name != whisper_model_size_option: # Перевіряємо, чи змінився розмір
                with st.spinner(f"Завантаження моделі Whisper '{whisper_model_size_option}'... Це може зайняти деякий час."):
                    whisper_model_instance = load_whisper_model(whisper_model_size_option) # Перезавантажуємо, якщо змінився розмір
                    if whisper_model_instance is None:
                        st.error("Не вдалося завантажити модель Whisper. Обробка неможлива.")
            
            if whisper_model_instance is not None:
                with st.status("Початок обробки...", expanded=True) as status:
                    results = process_media(
                        uploaded_file, whisper_model_instance, whisper_language_option, 
                        whisper_task_option, target_m2m_lang_choice, status
                    )
                    st.session_state.transcription_processing_result = {
                        "text_whisper": results[0] if results[0] and "Помилка" not in results[0] else "",
                        "txt_path_whisper": results[1],
                        "detected_language_whisper": results[2],
                        "segments_whisper": results[3],
                        "auto_translated_text_m2m": results[4] if results[4] and "Помилка" not in results[4] else None,
                        "auto_translation_m2m_path": results[5]
                    }
                    logging.info(f"Результати обробки збережено в session_state: {st.session_state.transcription_processing_result}")

        st.markdown("---")
        st.markdown("## 📜 Результати обробки:")
        
        res = st.session_state.transcription_processing_result
        
        st.markdown("#### Текст розшифровки (Whisper):")
        st.text_area(
            label="Розшифрований текст",
            value=res.get("text_whisper", ""),
            height=150,
            disabled=True,
            key="whisper_text_output_area"
        )
        if res.get("txt_path_whisper") and os.path.exists(res["txt_path_whisper"]):
            try:
                with open(res["txt_path_whisper"], "rb") as f:
                    st.download_button(
                        label="⬇️ Завантажити розшифровку (TXT)",
                        data=f,
                        file_name=os.path.basename(res["txt_path_whisper"]),
                        mime="text/plain",
                        key="download_whisper_txt"
                    )
            except Exception as e:
                st.error(f"Не вдалося підготувати TXT файл розшифровки для завантаження: {e}")

        if res.get("auto_translated_text_m2m"):
            st.markdown("#### Автоматичний переклад (M2M100):")
            st.text_area(
                label="Перекладений текст (M2M100)",
                value=res.get("auto_translated_text_m2m", ""),
                height=150,
                disabled=True,
                key="m2m_auto_translation_output_area"
            )
            if res.get("auto_translation_m2m_path") and os.path.exists(res["auto_translation_m2m_path"]):
                try:
                    with open(res["auto_translation_m2m_path"], "rb") as f_trans:
                        st.download_button(
                            label="⬇️ Завантажити авто-переклад (TXT)",
                            data=f_trans,
                            file_name=os.path.basename(res["auto_translation_m2m_path"]),
                            mime="text/plain",
                            key="download_m2m_auto_translation_txt"
                        )
                except Exception as e:
                    st.error(f"Не вдалося підготувати TXT файл авто-перекладу для завантаження: {e}")
        elif res.get("auto_translated_text_m2m") is not None and "Помилка" in res.get("auto_translated_text_m2m", ""):
             st.error(f"Помилка автоматичного перекладу: {res.get('auto_translated_text_m2m')}")


    with tab2:
        st.markdown("### 🌐 Ручний переклад розшифрованого тексту (M2M100)")
        original_text_for_manual_translation = st.session_state.transcription_processing_result.get("text_whisper", "")

        if not original_text_for_manual_translation:
            st.info
            st.info("Спочатку виконайте розшифровку на вкладці 'Розшифровка та Авто-переклад', щоб отримати текст для ручного перекладу.")
        else:
            st.markdown("#### Текст для перекладу (з розшифровки Whisper):")
            st.text_area(
                label="Оригінальний текст",
                value=original_text_for_manual_translation,
                height=150,
                disabled=True,
                key="manual_translation_source_text"
            )

            manual_translation_languages = get_supported_languages_m2m100()
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                manual_source_lang = st.selectbox(
                    "Мова оригіналу (для M2M100):",
                    options=list(manual_translation_languages.keys()),
                    format_func=lambda x: f"{manual_translation_languages[x]} ({x})",
                    # Спробувати встановити на основі detected_language_whisper або 'en'
                    index=list(manual_translation_languages.keys()).index(
                        st.session_state.transcription_processing_result.get("detected_language_whisper", "en").split('_')[0]
                    ) if st.session_state.transcription_processing_result.get("detected_language_whisper", "en").split('_')[0] in manual_translation_languages else (list(manual_translation_languages.keys()).index("en") if "en" in manual_translation_languages else 0),
                    key="manual_source_lang_select"
                )
            with col_m2:
                manual_target_lang = st.selectbox(
                    "Мова перекладу (для M2M100):",
                    options=list(manual_translation_languages.keys()),
                    format_func=lambda x: f"{manual_translation_languages[x]} ({x})",
                    index=list(manual_translation_languages.keys()).index("uk") if "uk" in manual_translation_languages else 0,
                    key="manual_target_lang_select"
                )
            
            manual_translate_button = st.button("🌐 Перекласти текст (M2M100)", key="manual_translate_button")

            if 'manual_translation_result_text' not in st.session_state:
                st.session_state.manual_translation_result_text = None
            if 'manual_translation_result_path' not in st.session_state:
                st.session_state.manual_translation_result_path = None

            if manual_translate_button:
                with st.spinner("Виконується ручний переклад..."):
                    translated_text, translation_path = translate_with_m2m100(
                        original_text_for_manual_translation, manual_source_lang, manual_target_lang
                    )
                    st.session_state.manual_translation_result_text = translated_text
                    st.session_state.manual_translation_result_path = translation_path
                    if "Помилка" in translated_text :
                        st.error(f"Помилка ручного перекладу: {translated_text}")
                    else:
                        st.success("Ручний переклад завершено!")
            
            if st.session_state.manual_translation_result_text and "Помилка" not in st.session_state.manual_translation_result_text:
                st.markdown("#### Результат ручного перекладу:")
                st.text_area(
                    label="Перекладений текст",
                    value=st.session_state.manual_translation_result_text,
                    height=150,
                    disabled=True,
                    key="manual_translation_output_area"
                )
                if st.session_state.manual_translation_result_path and os.path.exists(st.session_state.manual_translation_result_path):
                    try:
                        with open(st.session_state.manual_translation_result_path, "rb") as f_manual_trans:
                            st.download_button(
                                label="⬇️ Завантажити ручний переклад (TXT)",
                                data=f_manual_trans,
                                file_name=os.path.basename(st.session_state.manual_translation_result_path),
                                mime="text/plain",
                                key="download_manual_translation_txt"
                            )
                    except Exception as e:
                        st.error(f"Не вдалося підготувати файл ручного перекладу для завантаження: {e}")
            elif st.session_state.manual_translation_result_text: # Якщо є текст, але це помилка
                 st.error(f"Помилка ручного перекладу: {st.session_state.manual_translation_result_text}")


    st.markdown("---")
    st.markdown("## 📁 Управління файлами")
    output_files = get_output_files()
    if output_files:
        st.write(f"Згенеровані файли ({len(output_files)}):")
        with st.container(height=200): # Обмежуємо висоту контейнера
            for file_path in output_files:
                col_f1, col_f2, col_f3 = st.columns([5, 2, 1])
                file_name = os.path.basename(file_path)
                try:
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    with col_f1:
                        st.text(file_name)
                    with col_f2:
                        st.text(f"{file_size:.1f} KB")
                    with col_f3:
                        with open(file_path, "rb") as fp_down:
                            st.download_button(
                                label="⬇️",
                                data=fp_down,
                                file_name=file_name,
                                mime="application/octet-stream",
                                key=f"download_list_{file_name}"
                            )
                except FileNotFoundError:
                    st.text(f"{file_name} (файл не знайдено, можливо, видалено)")
                except Exception as e_file:
                    st.text(f"{file_name} (помилка: {e_file})")
        
        if st.button("🗑️ Очистити всі згенеровані файли"):
            delete_all_output_files()
            st.success("Всі згенеровані файли видалено.")
            # Очищення стану для відображення
            st.session_state.transcription_processing_result = {
                "text_whisper": "", "txt_path_whisper": None, "detected_language_whisper": None, 
                "segments_whisper": None, "auto_translated_text_m2m": None, "auto_translation_m2m_path": None
            }
            st.session_state.manual_translation_result_text = None
            st.session_state.manual_translation_result_path = None
            st.experimental_rerun()
    else:
        st.info("Немає доступних згенерованих файлів.")
    
    st.markdown("### 💾 Інформація про диск")
    try:
        total_output_size = 0
        if os.path.exists(OUTPUT_DIR):
            for item in os.listdir(OUTPUT_DIR):
                item_path = os.path.join(OUTPUT_DIR, item)
                if os.path.isfile(item_path):
                    total_output_size += os.path.getsize(item_path)
        
        disk = shutil.disk_usage("/")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Використано (зген. файли)", f"{total_output_size / (1024*1024):.2f} MB")
        with col_d2:
            st.metric("Вільно на диску", f"{disk.free / (1024*1024*1024):.1f} GB")
        with col_d3:
            st.metric("Всього на диску", f"{disk.total / (1024*1024*1024):.1f} GB")
    except Exception as e:
        st.error(f"Не вдалося отримати інформацію про диск: {e}")
    
    with st.expander("ℹ️ Інформація про додаток"):
        st.markdown("""
        ### 🎤 Розшифровка та переклад аудіо/відео
        
        Цей додаток дозволяє:
        1. 📝 **Розшифровувати** аудіо та відео файли за допомогою моделі Whisper.
        2. 🌐 **Автоматично перекладати** розшифрований текст на обрану мову за допомогою M2M-100.
        3. 🔄 **Вручну перекладати** розшифрований текст на вкладці "Ручний Переклад".
        
        **Підтримувані формати файлів для завантаження:**
        - Аудіо: `.wav`, `.mp3`, `.mpga`
        - Відео: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.mpeg`, `.mpeg4` 
          (з відео буде автоматично витягнуто аудіодоріжку)
        
        **Використовувані технології:**
        - [Whisper](https://openai.com/research/whisper) від OpenAI для розшифровки.
        - [M2M-100](https://huggingface.co/facebook/m2m100_418M) від Facebook/Meta для перекладу.
        
        **Про обмеження:**
        - Більші файли та більші моделі Whisper потребують більше часу та ресурсів.
        - Якість залежить від якості аудіо та обраної моделі.
        - Переклад може мати неточності.
        
        **Порада:** Для швидкої обробки використовуйте моделі Whisper `tiny` або `base`. Для кращої якості – `small`, `medium` або `large`.
        """)
    
    st.markdown("---")
    footer_col1, footer_col2 = st.columns(2)
    with footer_col1:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gpu_info = "Доступний" if torch.cuda.is_available() else "Недоступний"
        st.write(f"🕒 {current_time} | 🖥️ GPU: {gpu_info}")
    with footer_col2:
        st.write("© 2023-2024 | Версія 2.0.0 | Зроблено в Україні 🇺🇦")

if __name__ == "__main__":
    main()
