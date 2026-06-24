python build/build.py build --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt --local_xla_path=../xla --editable --cuda_version=12.9.2
pip install dist/*.whl  # installs jaxlib (includes XLA)
pip install -e . # installs jax

# jax depends on cudnn and nccl, there should be no conflict with locally built torch.
pip install "nvidia-cudnn-cu12>=9.8,<10" "nvidia-nccl-cu12"
