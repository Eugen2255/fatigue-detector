#!/bin/bash

OUTPUT="take.txt"
> "$OUTPUT"

# Функция для удаления комментариев
strip_comments() {
    local file="$1"
    local ext="${file##*.}"

    case "$ext" in
        py|sh|yaml|yml|cfg|ini|toml|txt|md)
            # Удаляет строки, начинающиеся с # (с учетом пробелов)
            sed 's/^\s*#.*$//' "$file"
            ;;
        cpp|h|hpp|c|cc|cxx|js|ts|java|cs)
            # Удаляет // комментарии
            sed 's|^\s*//.*$||' "$file"
            ;;
        *)
            # Для остальных файлов выводим как есть
            cat "$file"
            ;;
    esac
}

git ls-files | while read -r file; do
    if [ -f "$file" ]; then
        echo "--- FILE: $file ---" >> "$OUTPUT"
        strip_comments "$file" >> "$OUTPUT"
        echo -e "\n\n" >> "$OUTPUT"
    fi
done

echo "Готово: $OUTPUT"