import multiprocessing as mp

from run.lifesim.lifesim_run_multiple import main as lifesim_main
from run.hwo.hwo_run_multiple import main as hwo_main

if __name__ == '__main__':
    mp.set_start_method('spawn')
    lifesim_main()
    hwo_main()