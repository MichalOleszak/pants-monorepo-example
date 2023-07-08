files(
    name="module_files",
    sources=["**/*"],
)

python_requirements(
    name="reqs",
    source="requirements.txt",
    resolve="mnist",
)

docker_image(
    name="train_mnist",
    dependencies=["mnist:module_files"],
    registries=["docker.io"],
    repository="michaloleszak/mnist",
    image_tags=["latest", "{build_args.SHORT_SHA}"],
)