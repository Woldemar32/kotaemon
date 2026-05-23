.PHONY: install install-gpu run dev

install:
	uv venv .venv --python 3.11
	uv pip install -r requirements_gerageragera39.txt
	uv pip install -e ".[dev]"

install-gpu: install
	uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

run:
	.venv/Scripts/python app.py
	# .venv/bin/python app.py      # Linux/Mac

dev:
	.venv/Scripts/python app.py --reload
