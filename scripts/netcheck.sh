#!/bin/bash
# network connectivity probe (run inside WSL)
for u in https://huggingface.co https://hf-mirror.com https://pypi.org/simple/ https://pypi.tuna.tsinghua.edu.cn/simple/ https://mirrors.aliyun.com/pypi/simple/ https://github.com https://download.pytorch.org/whl/cu124/; do
  r=$(curl -s -o /dev/null -m 15 -w "%{http_code} %{time_total}s" "$u")
  echo "$u -> $r"
done
