run-dev:
	PYTHONPATH=src uvicorn main:app --reload --log-config=src/log_conf.yaml
