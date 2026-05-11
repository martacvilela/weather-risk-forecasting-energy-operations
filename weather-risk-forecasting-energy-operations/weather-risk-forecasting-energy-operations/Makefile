.PHONY: install train test app clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest

train:
	PYTHONPATH=src python -m weather_risk.train

test:
	PYTHONPATH=src pytest -q

app:
	streamlit run streamlit_app/app.py

clean:
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
