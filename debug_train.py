"""
本地调试入口,等价于:
    bash tools/dist_train.sh projects/configs/maptrv2/maptrv2_xd_raster_boundary.py 1

区别:不通过 torch.distributed.launch 拉起子进程,而是在当前进程内
模拟单机单卡的分布式环境变量,因此 IDE 断点可以直接生效。
用法:在 IDE 中以调试模式(Debug)直接运行本文件即可。
"""

import os
import sys
import runpy

# ---------- 配置区 ----------
CONFIG = 'projects/configs/maptrv2/maptrv2_xd_raster_boundary.py'
PORT = 28509        # 与 dist_train.sh 默认端口一致
WORK_DIR = None     # None 则沿用 config 里的 work_dir
EXTRA_ARGS = []     # 追加参数,例如 ['--no-validate']
DISTRIBUTED = True  # True: 进程内模拟单卡 DDP(与 dist_train.sh 行为一致)
                    # False: 完全非分布式单进程(排查 DDP 相关问题以外更简单)
# ----------------------------


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)             # 保证 config / work_dirs 等相对路径正确
    sys.path.insert(0, root)   # 等价于 dist_train.sh 中的 PYTHONPATH 设置

    if DISTRIBUTED:
        # 模拟 torch.distributed.launch 单机单卡时注入的环境变量
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(PORT)

    train_py = os.path.join(root, 'tools', 'train.py')
    config = os.path.join(root, CONFIG)

    sys.argv = [train_py, config, '--deterministic'] + EXTRA_ARGS
    if WORK_DIR is not None:
        sys.argv += ['--work-dir', WORK_DIR]
    if DISTRIBUTED:
        sys.argv += ['--launcher', 'pytorch']
    # else: launcher 默认 'none',即普通单进程训练

    # 以 __main__ 身份原地执行 tools/train.py,与命令行行为完全一致
    runpy.run_path(train_py, run_name='__main__')


if __name__ == '__main__':
    main()