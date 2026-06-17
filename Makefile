.PHONY: install run test docker
install:
	pip install -r requirements.txt
run:
	python main.py
test:
	pytest -q
docker:
	docker build -t student-success-analytics . && docker run --rm student-success-analytics
