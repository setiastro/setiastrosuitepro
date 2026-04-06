# Seti Astro Suite Pro (SASpro)

**Author:** Franklin Marek

**Website:** [www.setiastro.com](https://www.setiastro.com)

## Contributors

The following individuals have made substantial contributions to the development of Seti Astro Suite Pro. Their work spans core architecture, tooling, platform support, localization, and documentation.

### [Fabio Tempera](https://github.com/Ft2801)

- Complete structural refactoring of `setiastrosuitepro.py` (20,000+ lines), including elimination of duplicated logic and consolidation of responsibilities across the entire codebase
- Implementation of the AstroSpikes tool, Texture and Clarity processing module, real-time system resource monitor, and application usage statistics panel
- Addition of more than ten interface language translations, covering Arabic, German, Spanish, French, Hindi, Italian, Japanese, Portuguese, Russian, Swahili, Ukrainian, and Chinese
- Design and implementation of UI layout components, startup splash window, and application initialization optimizations
- Introduction of lazy import strategies, module-level caching mechanisms, and utility function libraries to reduce startup time and memory footprint
- Authorship of this README and project documentation

### [Joaquin Rodriguez](https://github.com/jrhuerta)

- Migration of the project dependency system to [Poetry](https://python-poetry.org/), introducing `pyproject.toml` as the canonical dependency manifest and establishing a reproducible build environment

### [Tim Dicke](https://github.com/dickett)

- Development of native installers for Windows and macOS
- Authorship and ongoing maintenance of macOS-specific Wiki installation instructions
- Application testing across platforms and resolution of minor defects

### [Michael Lev](https://github.com/MichaelLevAstro)

- Addition of the Hebrew (`he`) interface translation

### [Andrew Witwicki](https://github.com/awitwicki)

- Addition of the Ukrainian (`uk`) interface translation

## Overview

Seti Astro Suite Pro (SASpro) is a comprehensive astrophotography processing toolkit designed to cover the full image processing pipeline — from raw calibration frames through final export. The application targets both amateur astrophotographers and professional researchers, providing a graphical user interface suitable for interactive workflows as well as scripting interfaces for batch automation.

The project is built around three principal objectives:

- Produce repeatable, high-quality results through well-defined, algorithm-driven processing pipelines
- Expose advanced image processing capabilities through an accessible and consistent graphical interface
- Maintain a modular, extensible codebase that accommodates community contributions without sacrificing architectural coherence

SASpro is distributed as donationware. It is free to use, with an optional suggested donation to support continued development.

## Features

### Image Format Support

SASpro supports a broad range of astronomical and general-purpose image formats:

- **FITS** — the standard format for scientific astronomical imaging
- **XISF** — PixInsight's native extensible image serialization format
- **TIFF** — high-bit-depth raster format widely used in post-processing workflows
- **RAW** — native sensor data from digital cameras (via appropriate decoding libraries)
- **PNG / JPEG** — standard raster formats for preview and export purposes

### Calibration and Pre-Processing

- Master frame generation and application: bias, dark, and flat calibration pipelines
- Debayering (CFA demosaicing) for color sensor data
- Pedestal removal and sensor-level corrections
- SER file batch import and stacking for planetary imaging sequences

### Registration and Stacking

- Automated star detection and feature-based image registration
- Multi-frame stacking with configurable rejection algorithms
- Comet stacking mode with dual-target alignment support
- Live stacking interface for real-time session accumulation
- Mosaic composition tooling for multi-panel wide-field imaging

### Photometry and Astrometry

- Aperture photometry with configurable aperture and annulus parameters
- Signal-to-noise ratio (SNR) measurement tool
- Plate solving integration for accurate world coordinate system (WCS) assignment
- GAIA catalog downloader for reference star data
- Minor body catalog integration for asteroid and comet identification
- Exoplanet transit detection utility
- Supernova and asteroid hunting tool

### Color Processing

- Spectrophotometric color calibration (SPCC)
- Narrowband-to-RGB combination with normalization tools
- Background neutralization and gradient removal (including GraXpert integration)
- Selective color and selective luminance adjustment
- White balance correction (including star-based white balance)
- Green channel noise suppression (SCNR)
- RGB channel extraction, combination, and alignment

### Stretching and Tone Mapping

- Generalized Hyperbolic Stretch (GHS) with interactive preview
- Statistical stretch and auto-stretch utilities
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Histogram transformation with interactive curve manipulation
- Wavescale HDR processing pipeline
- Multiscale decomposition for detail and layer separation
- Frequency separation for independent large-scale and fine-detail processing

### AI-Assisted Processing

- CosmicClarity engine suite: denoising, sharpening, dark star recovery, super-resolution, and satellite trail removal
- Syqon Prism denoising engine
- Syqon Parallax sharpening engine
- Syqon NafNet starless separation engine
- AI-driven aberration correction module
- Multi-frame deconvolution (MFDeconv) with CuDNN and sport-mode variants
- Morphological star reduction and halo suppression

### Masking and Layers

- Interactive mask creation, application, and removal
- Layer-based compositing system with dockable layer panel
- Clone stamp and blemish removal tools
- Image combine with configurable blend modes

### Utilities and Tools

- FITS header viewer and metadata editor
- PSF (Point Spread Function) viewer and analysis
- Isophote fitting for galaxy morphology analysis
- Finder chart generation
- ACV curve export compatibility
- Pixel math scripting interface
- Batch conversion and batch renaming utilities
- History explorer for processing session review
- AstroBin export integration
- Keyboard shortcut manager
- Application statistics panel
- Real-time system resource monitor (CPU, memory, GPU)
- Blink comparator for image-to-image comparison

### Internationalization

The interface is fully localized in the following languages, with compiled Qt translation files (`.qm`) provided for each:

| Language | Code |
|---|---|
| Arabic | `ar` |
| German | `de` |
| Spanish | `es` |
| French | `fr` |
| Hebrew | `he` |
| Hindi | `hi` |
| Italian | `it` |
| Japanese | `ja` |
| Portuguese | `pt` |
| Russian | `ru` |
| Swahili | `sw` |
| Ukrainian | `uk` |
| Chinese (Simplified) | `zh` |

## Architecture and Project Layout

The repository follows a modular package structure. The primary application code resides under `src/setiastro/saspro/`, with supporting tooling and configuration at the repository root.

### Top-Level Directory Overview

| Path | Description |
|---|---|
| `src/setiastro/saspro/` | Core application package containing all processing modules, GUI components, engines, and utilities |
| `src/setiastro/data/` | Bundled data assets including the main FITS dataset and all CSV catalogs |
| `src/setiastro/images/` | All application icons, toolbar images, and UI graphic assets |
| `src/setiastro/qml/` | QML components, including the system resource monitor overlay |
| `ops/` | Repository-level tooling scripts for build management and dependency export |
| `config/` | Packaging configuration files, including PyInstaller `.spec` files |
| `build/` | Distribution and packaging output scripts |
| `logs/` | Runtime log output directory (development mode) |

### Internal Package Structure (`src/setiastro/saspro/`)

#### `gui/`

The graphical interface layer, organized using a mixin-based composition pattern to isolate concerns across the main window:

| Module | Responsibility |
|---|---|
| `main_window.py` | Central application window; coordinates all docks, toolbars, and the MDI workspace |
| `statistics_dialog.py` | Application usage statistics dialog |
| `mixins/dock_mixin.py` | Dockable panel management |
| `mixins/file_mixin.py` | File open, save, and recent file handling |
| `mixins/geometry_mixin.py` | Window geometry persistence and restoration |
| `mixins/header_mixin.py` | FITS header display integration |
| `mixins/mask_mixin.py` | Mask workflow integration |
| `mixins/menu_mixin.py` | Menu bar construction and action wiring |
| `mixins/theme_mixin.py` | Application theme and stylesheet management |
| `mixins/toolbar_mixin.py` | Toolbar layout and tool action registration |
| `mixins/update_mixin.py` | Update check and notification handling |
| `mixins/view_mixin.py` | Zoom, pan, and view state management |

#### `widgets/`

Reusable UI components shared across processing dialogs:

| Module | Responsibility |
|---|---|
| `common_utilities.py` | Shared widget helper functions |
| `graphics_views.py` | Custom `QGraphicsView` subclasses for image display |
| `image_utils.py` | Image conversion and display pipeline utilities |
| `preview_dialogs.py` | Base classes for before/after preview dialog patterns |
| `resource_monitor.py` | Real-time CPU, memory, and GPU utilization widget |
| `spinboxes.py` | Extended numeric input controls |
| `themed_buttons.py` | Consistently styled button components |
| `wavelet_utils.py` | Wavelet decomposition utilities shared across processing tools |
| `minigame/` | Embedded browser-based minigame (HTML/CSS/JS) |

#### `cosmicclarity_engines/`

Inference engine wrappers for the CosmicClarity AI model family:

| Engine | Function |
|---|---|
| `benchmark_engine.py` | Hardware performance benchmarking |
| `darkstar_engine.py` | Dark star recovery and enhancement |
| `denoise_engine.py` | AI-based noise suppression |
| `satellite_engine.py` | Satellite and aircraft trail detection and removal |
| `sharpen_engine.py` | AI-based image sharpening |
| `superres_engine.py` | Super-resolution upscaling |

#### `denoise_engines/` / `sharpen_engines/` / `starless_engines/`

Additional AI inference engines based on the Syqon model family:

| Engine | Function |
|---|---|
| `syqon_prism_engine.py` | Alternative denoising pipeline |
| `syqon_parallax_engine.py` | Alternative sharpening pipeline |
| `syqon_nafnet_engine.py` | Star removal for starless processing workflows |

#### `imageops/`

Low-level image processing algorithms operating directly on array data:

| Module | Function |
|---|---|
| `stretch.py` | Core stretching algorithms |
| `narrowband_normalization.py` | Narrowband channel normalization |
| `scnr.py` | Selective color noise reduction |
| `serloader.py` | SER video format frame loader |
| `starbasedwhitebalance.py` | Star-color-based white balance computation |
| `mdi_snap.py` | MDI subwindow snapping geometry utilities |

#### `translations/`

Qt Linguist translation sources (`.ts`) and compiled binaries (`.qm`) for all supported interface languages, along with Python-side translation dictionaries and integration tooling.

#### `ops/`

Application-level operational components:

| Module | Function |
|---|---|
| `settings.py` | Persistent user preferences and configuration |
| `commands.py` | Undo/redo command objects |
| `command_runner.py` | Command execution and history stack |
| `scripts.py` | Script execution runtime |
| `script_editor.py` | Integrated script editor dialog |
| `benchmark.py` | System benchmark orchestration |

#### `legacy/`

Retained modules providing backward compatibility:

| Module | Function |
|---|---|
| `image_manager.py` | Legacy image slot management interface |
| `xisf.py` | XISF format reader/writer |
| `numba_utils.py` | Legacy Numba JIT utility functions |

### Notable Root-Level Files

| File | Description |
|---|---|
| `setiastrosuitepro.py` | Primary application entry point for development execution |
| `pyproject.toml` | Poetry-managed dependency manifest and project metadata |
| `requirements.txt` | pip-compatible dependency list generated from `pyproject.toml` |
| `poetry.lock` | Locked dependency resolution for reproducible environments |
| `create_dmg.sh` | Shell script for macOS DMG image creation |
| `update_saspro.sh` | Application update script |
| `updates.json` | Update manifest consumed by the in-application update checker |
| `numba_runtime_hook.py` | PyInstaller runtime hook for Numba JIT initialization |
| `profile_init.py` | Startup profiling utility for performance diagnostics |
| `PUBLISHING.md` | Internal release and publishing procedures |

## Repository Structure

```
setiastrosuitepro/
|-- ops/
|   |-- __init__.py
|   |-- export_requirements.py
|   |-- prime_matplotlib_cache.py
|   `-- write_build_info.py
|-- src/
|   `-- setiastro/
|       |-- data/
|       |   |-- catalogs/
|       |   |   |-- astrobin_filters.csv
|       |   |   |-- astrobin_filters_page1_local.csv
|       |   |   |-- cali2.csv
|       |   |   |-- cali2color.csv
|       |   |   |-- celestial_catalog.csv
|       |   |   |-- detected_stars.csv
|       |   |   |-- fits_header_data.csv
|       |   |   |-- List_of_Galaxies_with_Distances_Gly.csv
|       |   |   `-- updated_celestial_catalog.csv
|       |   `-- SASP_data.fits
|       |-- images/
|       |   `-- [application icons and UI graphics]
|       |-- qml/
|       |   `-- ResourceMonitor.qml
|       |-- saspro/
|       |   |-- _generated/
|       |   |-- cosmicclarity_engines/
|       |   |-- denoise_engines/
|       |   |-- gui/
|       |   |   `-- mixins/
|       |   |-- imageops/
|       |   |-- legacy/
|       |   |-- ops/
|       |   |-- sharpen_engines/
|       |   |-- starless_engines/
|       |   |-- syqon_model/
|       |   |-- syqon_parallax_model/
|       |   |-- syqon_prism_model/
|       |   |-- translations/
|       |   |-- widgets/
|       |   |   `-- minigame/
|       |   |-- __init__.py
|       |   |-- __main__.py
|       |   `-- [application modules]
|       `-- __init__.py
|-- .gitattributes
|-- .gitignore
|-- create_dmg.sh
|-- LICENSE
|-- numba_runtime_hook.py
|-- poetry.lock
|-- pyproject.toml
|-- requirements.txt
|-- setiastrosuitepro.py
|-- update_saspro.sh
|-- updates.json
`-- README.md
```

## Development Setup

The following procedure describes how to establish a local development environment from source. The instructions apply to all supported platforms; platform-specific commands are noted where they differ.

### Prerequisites

- Python 3.10 or later
- `pip` 23.0 or later (upgraded during setup)
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/setiastro/setiastrosuitepro.git
cd setiastrosuitepro
```

### Step 2 — Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is restricted by the system policy, adjust the execution policy for the current process:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\Activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Launch the Application

```bash
python setiastrosuitepro.py
```

## Dependency Management

This project uses [Poetry](https://python-poetry.org/) as its authoritative dependency management tool. The `pyproject.toml` file defines all runtime and development dependencies. The `requirements.txt` file is a derived artifact generated from `pyproject.toml` to ensure compatibility with standard `pip`-based workflows.

### For Contributors and Maintainers

All dependency additions or modifications must be made through `pyproject.toml`. After modifying dependencies, regenerate `requirements.txt` using one of the following methods:

**Using the project export script:**
```bash
poetry run python ops/export_requirements.py
```

**Using Poetry directly:**
```bash
poetry export -f requirements.txt --without-hashes --without dev -o requirements.txt
```

Commit both `pyproject.toml` and the regenerated `requirements.txt` together in the same commit.

### For End Users

Standard `pip` installation from `requirements.txt` remains fully supported and requires no knowledge of Poetry:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file is maintained in a current state at all times.

## Data and Catalogs

All reference data and catalog files are located under `src/setiastro/data/`:

- `SASP_data.fits` — the primary application dataset used by multiple processing modules. This file is large and is excluded from version control via `.gitignore` where applicable. It must be obtained separately or via the official distribution package.
- `data/catalogs/` — CSV-format astronomical catalogs including filter databases, celestial object catalogs, galaxy distance tables, and calibration reference data.

When adding custom catalogs, adhere to the CSV schema of the existing files and register the new catalog path using the `get_data_path()` helper function defined in `resources.py`.

## Logging

During development, the application writes log output to `saspro.log` within the `logs/` directory at the project root. In installed or packaged builds, log files are written to the appropriate per-platform user data directory.

Log file path resolution is implemented in `setiastrosuitepro.py`. Ensure the `logs/` directory exists and is writable when running in development mode to facilitate debugging.

## Testing

Unit and integration tests are placed in the `tests/` directory and executed using `pytest`.

**Install pytest:**
```bash
pip install pytest
```

**Run the test suite:**
```bash
pytest -q
```

Contributors are encouraged to include tests for new functionality and to verify that existing tests pass before submitting a pull request.

## Packaging

### macOS

The repository includes a PyInstaller specification file and a DMG creation script for macOS distribution:

```bash
pip install pyinstaller
pyinstaller --clean -y config/setiastrosuitepro_mac.spec
bash create_dmg.sh
```

Ensure the `.spec` file includes `src/setiastro/data/` and `src/setiastro/data/catalogs/` in its `datas` list so that all required assets are bundled correctly.

### Windows

Windows installer development is managed by Tim Dicke. Refer to the project Wiki for current Windows packaging instructions.

### Resource Path Resolution

When running as a packaged application, all data and asset paths are resolved through the `pro.resources` module. Use `get_resource_path()` and `get_data_path()` for any file access that must function correctly in both development and packaged contexts.

## Contributing

Contributions to SASpro are welcome. Please observe the following guidelines to maintain consistency and quality across the codebase.

### Workflow

1. Fork the repository on GitHub.
2. Create a dedicated feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Implement your changes. Keep commits atomic and scoped to a single logical change.
4. Include or update tests where applicable.
5. Regenerate `requirements.txt` if dependencies were modified.
6. Open a pull request against `main`, providing a clear description of the change, its motivation, and any relevant context.

### Code Standards

- Follow PEP 8 style conventions throughout.
- All new user-facing strings must be passed through the internationalization system (`i18n.py`) to support translation.
- New tools or processing dialogs should follow the established pattern of separating business logic from UI code.
- Avoid introducing large binary files into the repository. Data files should be referenced in `.gitignore` and distributed through appropriate channels.

For detailed repository-specific guidelines, refer to `CONTRIBUTING.md`.

## Troubleshooting

### Missing Data Files

If the application reports that a CSV or FITS file cannot be located, verify that the `src/setiastro/data/` and `src/setiastro/data/catalogs/` directories are present and populated. In packaged builds, confirm that these paths are included in the PyInstaller `.spec` configuration.

### Missing Dependencies

```bash
pip install -r requirements.txt
```

Ensure the virtual environment is activated before running this command.

### Log Write Permission Errors

Ensure the `logs/` directory exists and is writable by the current user. On some systems, elevated privileges may be required during the packaging phase.

### Reporting Bugs

If a reproducible defect is identified, open an issue on GitHub and attach the contents of `saspro.log`. Include the operating system version, Python version, and a minimal description of the steps required to reproduce the problem.

## License

Seti Astro Suite Pro is licensed under the **GNU General Public License v3.0**. See the `LICENSE` file at the repository root for the full license text.

## Acknowledgments

The authors gratefully acknowledge the open-source libraries and projects upon which SASpro depends, as well as the broader astrophotography community whose feedback has continuously shaped the direction of development.

## Contact and Links

| Resource | URL |
|---|---|
| Official Website | https://www.setiastro.com |
| Source Repository | https://github.com/setiastro/setiastrosuitepro |
| Issue Tracker | https://github.com/setiastro/setiastrosuitepro/issues |
