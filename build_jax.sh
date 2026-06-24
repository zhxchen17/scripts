python build/build.py build --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt --local_xla_path=../xla --editable --cuda_version=12.9.2
# installs jaxlib (includes XLA) [with-cuda] is the path taken by `pip install jax[cuda]` as well.
pip install -e dist/jaxlib -e dist/jax_cuda12_plugin[with-cuda] -e dist/jax_cuda12_pjrt  
pip install -e . # installs jax

# jax depends on cudnn and nccl, there should be no conflict with locally built torch.
pip install "nvidia-cudnn-cu12>=9.8,<10" "nvidia-nccl-cu12"
