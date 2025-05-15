# Master's Thesis in LaTeX

This repository contains a LaTeX template for a master's thesis, formatted according to the requirements of the Transport and Telecommunications Institute.

## 📁 Project Structure

```
.
├── .devcontainer/     # Development container configuration
├── .git/             # Git repository data
├── .vscode/          # VS Code settings
├── Src/              # Source files
│   ├── extra/        # Additional resources
│   ├── images/       # Image assets
│   ├── settings/     # LaTeX configuration files
│   └── templates/    # Document templates
└── main.tex          # Main document file
```

## 🛠️ Building the Project

### Prerequisites
- LaTeX distribution (TeX Live or MiKTeX)
- Make utility
- Git

### Compilation
To compile the project using Makefile:
```bash
git clone https://github.com/foxybbb/diploma-master.git
cd diploma-master
make
```

### Cleaning Build Files
To clean build files (except PDF):
```bash
make clean
```

## 📝 Document Structure

The thesis is organized into several sections:
- Introduction
- Theoretical Background
- Research Methodology
- Results and Discussion
- Conclusion
- References
- Appendices

## 🔧 Configuration

The project uses several LaTeX packages and configurations:
- `fontspec` for font management
- `natbib` for bibliography
- `hyperref` for hyperlinks
- `graphicx` for image handling
- And more...

## 📚 Bibliography

References are managed using BibTeX with the `plainnat` style. The bibliography file is located at `Src/refs.bib`.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Special thanks to all contributors and the LaTeX community for their support and resources.

## Работа с LaTeX

Пример компиляции проекта с помощью Makefile:
```shell
git clone https://github.com/foxybbb/diploma-master.git
cd diploma-master
make
```

Пример очистки сборочных файлов после компиляции (кроме PDF):
```shell
make clean
```

## Благодарности
