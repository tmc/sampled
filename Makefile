venv: requirements.in
venv:
	python3 -m venv venv
	venv/bin/pip install wheel pip-tools
	venv/bin/pip-compile requirements.in
	venv/bin/pip install -r requirements.txt
