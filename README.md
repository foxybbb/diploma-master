Дипломная работа бакалавра в LaTeX, оформленная в соответствии с требованиями Институт транспорта и связи.

# Структура

```
.
├── .devcontainer
├── .git
├── .vscode
├── Src
    ├── extra
    ├── images
    ├── settings
    └── templates

```


Каталог `.devcontainer/`

Каталог `.git/`

Каталог `Src/extra`

Каталог `Src/images`

Каталог `Src/settings`

Каталог `Src/templates`

## Работа с LaTeX

Пример компиляции проекта с помощью Makefile:
```shell
git clone https://github.com/foxybbb/diploma-bachelor.git
cd diploma-bachelor
make
```

Пример очистки сборочных файлов после компиляции (кроме PDF):
```shell
make clean
```

## Благодарности
