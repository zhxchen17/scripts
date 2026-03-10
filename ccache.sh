#!/usr/bin/env bash
# Install ccache from source and set up symlinks for compiler caching.

set -e

# Common util **functions** that can be sourced in other scripts.
function announce_step {
  echo
  echo "*************************"
  echo "$1"
  echo "*************************"
  echo
}

function mkdir_if_not_exists {
  if [[ ! -e "$1" ]]; then
      mkdir -p "$1"
  elif [[ ! -d "$1" ]]; then
      echo "$1 already exists but is not a directory, aborting..." 1>&2
      exit 1
  fi
}

function symlink_if_not_exists {
  if [[ ! -e "$2" ]]; then
      ln -s "$1" "$2"
  elif [[ ! -h "$2" ]]; then
      echo "$2 already exists but is not a symbolic link" 1>&2
  fi
}

CCACHE_HOME=$HOME/local/ccache

announce_step "Installing ccache from source"

mkdir_if_not_exists "$CCACHE_HOME"

# Download and build ccache from source
pushd "$CCACHE_HOME"
rm -rf ccache
git clone https://github.com/ccache/ccache.git
mkdir -p ccache/build
pushd ccache/build
cmake -DCMAKE_INSTALL_PREFIX="$CCACHE_HOME" \
  -DENABLE_TESTING=OFF \
  -DZSTD_FROM_INTERNET=ON \
  -DHIREDIS_FROM_INTERNET=ON \
  -DCMAKE_BUILD_TYPE=Release ..
make -j"$(nproc --ignore 1)" install
popd
popd

CCACHE_PATH="$CCACHE_HOME/bin/ccache"

# Set up symlinks so ccache intercepts compiler calls
announce_step "Setting up ccache symlinks"

mkdir_if_not_exists "$CCACHE_HOME/bin"
mkdir_if_not_exists "$CCACHE_HOME/lib"
mkdir_if_not_exists "$CCACHE_HOME/cuda"
symlink_if_not_exists "$CCACHE_PATH" "$CCACHE_HOME/bin/ccache"
symlink_if_not_exists "$CCACHE_HOME/bin/ccache" "$CCACHE_HOME/lib/cc"
symlink_if_not_exists "$CCACHE_HOME/bin/ccache" "$CCACHE_HOME/lib/c++"
symlink_if_not_exists "$CCACHE_HOME/bin/ccache" "$CCACHE_HOME/lib/gcc"
symlink_if_not_exists "$CCACHE_HOME/bin/ccache" "$CCACHE_HOME/lib/g++"
symlink_if_not_exists "$CCACHE_HOME/bin/ccache" "$CCACHE_HOME/cuda/nvcc"

# Set cache size to 25 GB
"$CCACHE_PATH" -M 25Gi

# Add ccache to PATH and set CUDA_NVCC_EXECUTABLE in ~/.bashrc
if ! grep -q "export PATH=$CCACHE_HOME/lib:$CCACHE_HOME/bin:\$PATH" ~/.bashrc; then
  echo "export PATH=$CCACHE_HOME/lib:$CCACHE_HOME/bin:\$PATH" >> ~/.bashrc
fi
if ! grep -q "export CUDA_NVCC_EXECUTABLE=$CCACHE_HOME/cuda/nvcc" ~/.bashrc; then
  echo "export CUDA_NVCC_EXECUTABLE=$CCACHE_HOME/cuda/nvcc" >> ~/.bashrc
fi

announce_step "Installation complete"
echo "ccache installed to $CCACHE_HOME"
echo "Remember to: source ~/.bashrc"
