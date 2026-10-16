PyTrajOpt uses [Poetry](https://python-poetry.org/) for robust dependency management and packaging. You can install the framework directly from GitHub or set it up locally for development.

### Option 1: Install Directly from GitHub via Pip

If you want to quickly install the package into your current Python environment without cloning the repository:

```bash
pip install git+https://github.com/sandeep026/pytrajopt.git

```

### Option 2: Install Directly from GitHub via Poetry

To add PyTrajOpt as a dependency in another Poetry-managed project directly from the remote repository:

```bash
poetry add git+https://github.com/sandeep026/pytrajopt.git

```

### Option 3: Local Installation for Development (Clone & Poetry)

If you want to run the provided examples or contribute to the source code:

1. **Clone the repository:**
```bash
git clone https://github.com/sandeep026/pytrajopt.git
cd pytrajopt

```

2. **Install dependencies:**
Poetry will automatically read the `pyproject.toml` configuration, resolve the dependencies (such as CasADi, Pydantic, NumPy, and Matplotlib), and set up a dedicated virtual environment:


```bash
poetry install

```

3. **Activate the virtual environment:**
```bash
poetry shell
```
