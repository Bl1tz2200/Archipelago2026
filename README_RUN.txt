ЗАПУСК ИЗ БРАУЗЕРНОГО VSCODE ДРОНА

1. Откройте браузерный VSCode дрона по адресу 192.168.1.126.
2. Загрузите и распакуйте каталог head_drone_integrated без изменения структуры apple_vision.
3. Откройте встроенный терминал VSCode и перейдите в каталог проекта:

   cd ~/head_drone_integrated

4. Проверка камеры и распознавания БЕЗ полёта:

   python3 head_drone.py

   Размеченный поток публикуется в /out_detection.

5. Полётный запуск только после проверки камеры и направления коррекции:

   python3 head_drone.py --fly

Параметры поиска по реальному полю:

   --field-width 6.0
   --field-height 6.0
   --lane-step 0.5
   --alt 1.5

Пример:

   python3 head_drone.py --fly --field-width 6.0 --field-height 6.0 --lane-step 0.5 --alt 1.5

Если дрон корректируется от цели, а не к ней:

   --image-x-sign 1 --image-y-sign 1

Папка apple_vision и её структура не изменялись.
