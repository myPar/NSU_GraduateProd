# Как построить

1. Требуется установить CMake (минимальная версия - 3.9). Инсталяторы можно загрузить с официального [сайта](https://cmake.org/download/).
2. Открываем командную строку в дирректории __keras2cpp_module__ и вводим следующие команды:
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```
3. После построения, исполняемый файл модуля будет располагаться в директории __build/Release/__. Его требуется переместить в директорию __keras2cpp_module__.

4. теперь можно запускать:
```bash
1D_2L_chart.exe
```