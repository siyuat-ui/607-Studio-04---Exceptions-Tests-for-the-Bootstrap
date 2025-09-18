# 607-Studio-04---Exceptions-Tests-for-the-Bootstrap

This is Siyuan Tang and Ziheng Wei's repo for Studio 04, STATS 607 001 FA 2025. Descriptions about this studio can be found [here](https://github.com/siyuat-ui/607-Studio-04---Exceptions-Tests-for-the-Bootstrap/blob/main/Studio%2004%20-%20Exceptions%20%26%20Tests%20for%20the%20Bootstrap.pdf).

Ziheng Wei is student A, who is responible for tests of bootstrap_ci and R_squared , and the implementation of bootstrap_sample. Siyuan Tang is student B, who is responsible for tests of bootstrap_sample , and the implementation of bootstrap_ci and R_squared. Both of them ran the tests, debugged, and added a statistical validation test that checks the bootstrap implementation against the known theoretical null distribution of R-squared.

## Quick start

0. Prerequisites
- Python 3.7 or higher
- pip package manager

1. Clone the repo

```bash
git clone https://github.com/siyuat-ui/607-Studio-04---Exceptions-Tests-for-the-Bootstrap.git
cd 607-Studio-04---Exceptions-Tests-for-the-Bootstrap
```

2. Create a virtual environment (recommended)

```bash
python3 -m venv boot_env

# On Windows
boot_env\Scripts\activate

# On macOS/Linux
source boot_env/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the test
```bash
pytest
```