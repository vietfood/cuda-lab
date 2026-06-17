How to connect to PrimeIntellect GPUs:

```bash
Host PrimeIntellect
    HostName <ip_address>
    User <user_name>
    IdentityFile ~/.ssh/prime
    IdentitiesOnly yes
```

How to connect with Github:

```bash
# generate key
ssh-keygen -t ed25519 -C "github"
# then copy key to deploy key by Github
```

Export CUDA:

```bash
# check nvcc
nvcc --version
# should show: /usr/local/cuda-<version>/bin/nvcc

# add path
cat >> ~/.bashrc <<'EOF'
export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
EOF

source ~/.bashrc
```