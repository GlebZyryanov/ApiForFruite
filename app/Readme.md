# Запуск
curl -X POST "http://localhost:8000/predict/" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@Путь до файла сканирования"

  В данном случае использовался файл из датасета фрукто, просто добавил прямую ссылку до него