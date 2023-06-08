init:
	pip install pre-commit==3.0.4
	pre-commit install --hook-type pre-commit --hook-type pre-push
	pip install pantsbuild.pants==2.15.1