# this is the default, check out mm_run for default values in argparse
# nsamps = 20000
# epochs = 100
# seqlen = 3
# batch_size = 32
# frame_size = 64
# ttRank = 64
# digit_size = 28
# speed = 5
# nx = [4, 16, 16, 4]
# nh = [2,  4,  4, 2]

python mm_run.py --batch_size=32 --sh 1024 --nh=[4,8,8,4] --nx=[4,16,16,4] --speed=5 --ttRank=64 