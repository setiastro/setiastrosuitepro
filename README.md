# Seti Astro Suite Pro (SASpro)

**Author:** Franklin Marek  
**Website:** [www.setiastro.com](http://www.setiastro.com)

---

## 📖 About
Seti Astro Suite Pro (SASpro) is an advanced, cross-platform astrophotography toolkit designed to streamline image processing, stacking, star analysis, and deep-sky imaging workflows. SASpro integrates powerful algorithms and a modern GUI to help both amateur and professional astronomers achieve high-quality results with minimal effort.

- **Key Features:**
  - Multi-format image support (FITS, TIFF, RAW, XISF, PNG, JPEG, etc.)
  - Automated calibration, stacking, and registration
  - Advanced star detection, photometry, and astrometry
  - White balance, background neutralization, and color calibration tools
  - AI-powered aberration correction and blemish removal
  - Batch processing and scripting capabilities
  - Interactive plotting and visualization (matplotlib, plotly, pyqtgraph)
  - Catalog integration (NGC, IC, Messier, custom CSVs)
  - Export to AstroBin and other platforms
  - Modular architecture for easy extension

SASpro is distributed as **donationware**. The software is free to use by anyone, but if you find it valuable, please consider a suggested donation of **$20** to support continued development.

👉 **Donate here:** [Donate to SASpro](https://www.setiastro.com/checkout/donate?donatePageId=65ae7e7bac20370d8c04c1ab)

---

## 🚀 Getting Started
# Seti Astro Suite Pro (SASpro)

**Author:** Franklin Marek  
**Website:** [www.setiastro.com](http://www.setiastro.com)

---

## Overview
Seti Astro Suite Pro (SASpro) is an advanced astrophotography toolkit for image calibration, stacking, registration, photometry, and visualization. It targets both amateur and professional users by offering a graphical user interface, batch processing scripts, and extension points for automation.

Key goals:
- Produce repeatable, high-quality astrophotography results
- Expose advanced algorithms through an approachable GUI
- Keep the codebase modular and extensible for community contributions

SASpro is distributed as donationware — free to use, with an optional suggested donation.

---

## Features
- Multi-format image support: FITS, XISF, TIFF, RAW, PNG, JPEG
- Calibration pipelines (bias/dark/flat), registration and stacking
- Star detection, aperture photometry, astrometry helpers
- Color calibration, white balance, background neutralization
- Blemish removal, aberration correction, and AI-based tools
- Batch processing and scripting interfaces
- Catalog support and CSV-based custom catalogs
- Export and integration helpers (e.g., AstroBin)

---

## Architecture and Project Layout
This project follows a modular layout. High-level modules and responsibilities:

- `pro/` - Primary application modules, UI, resources and business logic.
- `imageops/` - Image processing utilities and algorithms.
- `ops/` - Application-level operations, settings, and script runner.
- `scripts/` - Example scripts and small utilities that demonstrate automation.
- `data/` - Bundled data files and catalogs. (See `data/catalogs/` for CSV files.)
- `logs/` - Runtime logs produced during development or packaged runs.
- `config/` - Packaging specs and configuration files (e.g. `.spec`, update JSON).
- `build/` - Packaging and distribution scripts (e.g. `create_dmg.sh`).

Files of note:
- `setiastrosuitepro.py` - Application entrypoint used for development and direct runs.
- `setiastrosuitepro_mac.spec` - PyInstaller spec for macOS packaging (moved to `config/`).
- `SASP_data.fits` - Large dataset used by the app (moved to `data/`).
- `astrobin_filters.csv` and other CSV catalogs are under `data/catalogs/`.

Example tree (abridged):

```
setiastrosuitepro/
├── pro/
├── imageops/
├── ops/
├── scripts/
├── data/
│   ├── SASP_data.fits
│   └── catalogs/
│       ├── astrobin_filters.csv
│       └── celestial_catalog.csv
├── logs/
├── config/
├── build/
├── requirements.txt
├── setiastrosuitepro.py
└── README.md
```

---

## Quick Start — Development (Windows PowerShell example)
This section shows a minimal reproducible development setup using a Python virtual environment.

1. Open PowerShell and navigate to the project root.

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the application (development mode):

```powershell
python setiastrosuitepro.py
```

Notes:
- Use `Activate.bat` on Windows CMD, or `source .venv/bin/activate` on macOS/Linux.
- If you run into permission issues with `Activate.ps1`, you may need to change the execution policy temporarily:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

---

## Running a Packaged App
- Packagers like PyInstaller or similar are used to create distributables. See `config/setiastrosuitepro_mac.spec` and `build/create_dmg.sh` for packaging examples.
- When packaged, resources such as `SASP_data.fits` and `astrobin_filters.csv` are expected under the internal resources path. The application code resolves their paths using the `pro.resources` helpers.

---

## Data & Catalogs
- All CSV catalogs and reference data are in `data/catalogs/`.
- Large dataset files (e.g. `SASP_data.fits`) are in `data/` and are added to `.gitignore` when appropriate to avoid committing large binaries.
- If you add custom catalogs, follow the existing CSV schema and update `pro/resources.py` or use `get_data_path()` helper to resolve them.

---

## Logging
- During development the app writes `saspro.log` into the project `logs/` directory (or into per-platform user log directories when running installed builds).
- Log file location logic is implemented in `setiastrosuitepro.py` — keep `logs/` writeable for easier debugging.

---

## Testing
- Unit and integration tests can be created under a `tests/` directory and run with `pytest`.
- Example:

```powershell
pip install pytest
pytest -q
```

---

## Packaging Notes
- The repository contains a PyInstaller `.spec` file and helper scripts for macOS packaging.
- Typical packaging flow (example with PyInstaller):

```powershell
pip install pyinstaller
pyinstaller --clean -y config\setiastrosuitepro_mac.spec
```

Adjust spec paths to include `data/` and `data/catalogs/` as needed.

---

## Contributing
- Fork the repository and create a feature branch.
- Keep changes atomic and include tests when possible.
- Open a pull request describing the change and the reasoning.
- See `CONTRIBUTING.md` for repository-specific guidelines.

---

## Troubleshooting
- If the app cannot find a CSV or FITS file, verify the `data/` and `data/catalogs/` directories are present in the project root or that packaged resources are included during build.
- Common issues:
  - Missing dependencies: run `pip install -r requirements.txt`.
  - Permission errors when writing logs: ensure `logs/` is writeable or run with elevated privileges during packaging.

If you hit a reproducible bug, open an issue and attach the `saspro.log` file.

---

## License
- SASpro is licensed under **GNU GPLv3**. See `LICENSE` for details.

---

## Acknowledgments
Special thanks to the open-source projects and contributors used by SASpro.

---

## Contact & Links
- Website: https://www.setiastro.com
- Source: https://github.com/setiastro/setiastrosuitepro
- Issues: https://github.com/setiastro/setiastrosuitepro/issues

---
