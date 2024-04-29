# PyGPT

Use python3.12

## Setup Instructions

**MacOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**Install Dependencies**

With the virtual environment activated, install the project dependencies:

```bash
pip install -r requirements.txt
```

Let VSCode install the requirements for running the notebooks.

**CUDA**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```