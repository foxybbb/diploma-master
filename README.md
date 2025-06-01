# Master's Thesis in LaTeX

> **Template for Transport and Telecommunications Institute (TSI) master’s theses.**

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/foxybbb/diploma-master.git
cd diploma-master

# Build the thesis (PDF will appear in build/)
make

# Remove auxiliary files, keep the PDF
make clean
```

---

## 📁 Project Structure

```
.
├── Dockerfile       # Reproducible TeX Live toolchain
├── Makefile         # Build and clean targets
├── fonts/           # PT Sans & PT Mono used in the template
├── Materials/       # Source code, datasets and plots for experiments
│   ├── Code/            
│   ├── Experiments/     
│   ├── ROS2/            
│   └── SDK/             
├── Src/             # LaTeX sources
│   ├── chapters/        # One file per chapter
│   ├── images/          # Figures referenced in the thesis
│   ├── settings/        # Packages & project‑wide macros
│   ├── templates/       # Title, TOC & bibliography pages
│   ├── config.tex       # Global document options
│   ├── content.tex      # Includes the chapter files
│   └── refs.bib         # BibTeX database
├── .devcontainer/   # VS Code development container settings
├── .vscode/         # Editor tasks & LaTeX Workshop config
└── main.tex         # Entry point (inputs config.tex & content.tex)
```

### Directory overview

| Path                     | What it contains                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------- |
| `Dockerfile`             | Minimal image that installs TeX Live — ensures identical builds across platforms.     |
| `Makefile`               | Default target `make` compiles the thesis; `make clean` deletes auxiliary files.      |
| `fonts/`                 | OpenType fonts embedded in the final PDF to meet TSI style rules.                     |
| `Materials/Code/`        | Python/C++ scripts for LiDAR and radar data collection and processing.                |
| `Materials/Experiments/` | Raw CSV/binary captures, plotting scripts and resulting images.                       |
| `Materials/ROS2/`        | ROS2 nodes used during live sensor fusion tests.                                      |
| `Src/chapters/`          | `01_Introduction.tex`, `02_ResearchBackground.tex`, … — each chapter in its own file. |
| `Src/settings/`          | `packages.tex` (package list) and `preferences.tex` (custom macros & colours).        |
| `Src/templates/`         | Static pages: `titlepage.tex`, `tocpage.tex`, `bibpage.tex`.                          |
| `main.tex`               | Loads `config.tex`, then the actual content, then the template pages.                 |

---

## 📚 Bibliography

BibTeX entries reside in `Src/refs.bib` and are formatted with the `plainnat` style.

---

## 📄 License

Released under the **MIT License**. See `LICENSE` for details.

---

## 🤝 Contributing

Issues and pull requests are welcome — feel free to improve the template or report problems.

---

## 🙏 Acknowledgements

Thanks to the LaTeX community and previous TSI graduates whose work inspired this template.
