python build/build.py build --wheels=jax-cuda-plugin,jax-cuda-pjrt   --local_xla_path=../xla --bazel_options=--repo_env=LOCAL_CUDA_PATH="$(dirname $(dirname $(which nvcc)))"
