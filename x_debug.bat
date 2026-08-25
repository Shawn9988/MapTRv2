@echo off
ssh -L 5678:127.0.0.1:5678 huinian@10.130.21.244 "bash -i -c 'cd /nas/nfs/large-model/hn/code/RoGS && python train.py --config configs/local_ni.yaml'"
pause