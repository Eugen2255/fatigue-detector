import os
import subprocess
import sys

def get_git_files():
    try:
        result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True, check=True)
        return result.stdout.splitlines()
    except Exception:
        print("Ошибка: не удалось получить список файлов из git. Убедитесь, что вы в репозитории.")
        sys.exit(1)

def main():
    output_file = "take.txt"
    files = get_git_files()
    
    print(f"Найдено файлов: {len(files)}. Запись в {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for filepath in files:
            if not os.path.isfile(filepath):
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                out.write(f"--- FILE: {filepath} ---\n")
                out.write(content)
                # Гарантированный разделитель
                out.write("\n\n") 
            except Exception as e:
                print(f"⚠️ Пропуск {filepath}: {e}")

    print(f"✅ Готово: {output_file}")

if __name__ == "__main__":
    main()