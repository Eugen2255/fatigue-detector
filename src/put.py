import os
import sys

def main():
    dump_file = sys.argv[1] if len(sys.argv) > 1 else "take.txt"
    
    if not os.path.exists(dump_file):
        print(f"❌ Файл {dump_file} не найден")
        sys.exit(1)

    print(f"📖 Читаю {dump_file}...")
    
    with open(dump_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Разделяем по маркеру начала файла
    # Используем split, но первый элемент будет мусором до первого маркера
    parts = content.split("--- FILE: ")
    
    processed_count = 0
    
    # Пропускаем первую часть (она пустая или содержит текст до первого файла)
    for part in parts[1:]:
        # Ищем конец заголовка ---
        header_end = part.find(" ---\n")
        if header_end == -1:
            # Пробуем вариант без переноса строки (на всякий случай)
            header_end = part.find(" ---")
            if header_end == -1:
                continue
            
        filepath = part[:header_end]
        # Контент начинается после заголовка и переноса строки
        file_content = part[header_end + 5:] # 5 = len(" ---\n")
        
        # Убираем лишние переносы строк в конце, которые добавил скрипт экспорта
        # Но оставляем те, что были в оригинальном файле. 
        # Наш формат добавляет "\n\n" в конце каждого блока.
        if file_content.endswith("\n\n"):
            file_content = file_content[:-2]
        elif file_content.endswith("\n"):
            file_content = file_content[:-1]

        if not filepath:
            continue

        try:
            # Создаем директорию если нет
            dir_name = os.path.dirname(filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            print(f"✓ Обновлён: {filepath}")
            processed_count += 1
            
        except Exception as e:
            print(f"❌ Ошибка записи {filepath}: {e}")

    print(f"✅ Готово. Обработано файлов: {processed_count}")

if __name__ == "__main__":
    main()