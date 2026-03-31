# Portfolio Manager - Conda Environment Setup

## Environment Setup

This project uses a dedicated conda environment named `portman` with all required dependencies.

### Quick Start

1. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate portman
   ```

3. **Run the application:**
   ```bash
   python Portmanv1.py
   ```

### Environment Management

- **Activate environment:** `conda activate portman`
- **Deactivate environment:** `conda deactivate`
- **Remove environment:** `conda env remove -n portman`
- **Update environment:** `conda env update -f environment.yml`

### Dependencies

The `portman` environment includes:

- **Python 3.11**
- **Core libraries:** pandas, numpy, matplotlib
- **Financial data:** yfinance, quantstats, pyportfolioopt
- **Database:** duckdb
- **UI:** flet
- **File formats:** openpyxl, xlrd, html5lib, lxml, pyarrow, fastparquet

### Environment File

The `environment.yml` file defines the complete conda environment specification. It uses:
- `conda-forge` channel for high-quality packages
- Python 3.11 for stability and compatibility
- pip installations for packages not available via conda

### Verification

To verify the environment is working correctly:

```bash
conda activate portman
python -c "import flet, pandas, numpy, yfinance, duckdb, quantstats, matplotlib, pypfopt; print('All dependencies imported successfully!')"
```

This should print: "All dependencies imported successfully!"
