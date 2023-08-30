run:
	python3 icl

deps: requirements.txt
	pip install -r requirements.txt

test:
	pytest tests.py

clean:
	rm -rf __pycache__
