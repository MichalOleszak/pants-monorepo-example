files(
    name="module_files",
    sources=["**/*"],
)

python_sources(
    name="python",
    resolve="mnist",
    sources=["**/*.py"],
)

python_requirements(
    name="reqs",
    source="requirements.txt",
    resolve="mnist",
)

python_tests(
    name="tests",
    resolve="mnist",
    sources=["**/test_*.py"],
    dependencies=[
        "mnist:reqs",
        "mnist:module_files",
    ],
)

docker_image(
    name="train_mnist",
    dependencies=["mnist:module_files"],
    registries=["docker.io"],
    repository="michaloleszak/mnist",
    image_tags=["latest", "{build_args.SHORT_SHA}"],
)