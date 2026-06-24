python build/build.py build --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt   --local_xla_path=../xla --bazel_options=--repo_env=LOCAL_CUDA_PATH="$(dirname $(dirname $(which nvcc)))"
pip install dist/*.whl  # installs jaxlib (includes XLA)
