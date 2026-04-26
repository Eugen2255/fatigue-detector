#!/bin/bash

DUMP_FILE="${1:-take.txt}"

if [ ! -f "$DUMP_FILE" ]; then
    echo "Ошибка: файл $DUMP_FILE не найден"
    echo "Использование: $0 [путь_к_файлу_с_изменениями]"
    exit 1
fi

echo "Применяю изменения из $DUMP_FILE..."

current_file=""
in_content=0
content=""

while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" =~ ^---\ FILE:\ (.+)\ ---$ ]]; then
        # Сохраняем предыдущий файл если был
        if [ -n "$current_file" ] && [ -n "$content" ]; then
            dir=$(dirname "$current_file")
            [ -d "$dir" ] || mkdir -p "$dir"
            printf '%s' "$content" > "$current_file"
            echo "✓ Обновлён: $current_file"
        fi
        # Новый файл
        current_file="${BASH_REMATCH[1]}"
        content=""
        in_content=1
        continue
    fi
    
    if [ $in_content -eq 1 ]; then
        content+="$line"$'\n'
    fi
done < "$DUMP_FILE"

# Последний файл
if [ -n "$current_file" ] && [ -n "$content" ]; then
    dir=$(dirname "$current_file")
    [ -d "$dir" ] || mkdir -p "$dir"
    printf '%s' "$content" > "$current_file"
    echo "✓ Обновлён: $current_file"
fi

echo "✅ Готово"