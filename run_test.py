#!/usr/bin/env python

from __future__ import print_function

import os, glob, optparse, re, shutil, subprocess, sys, string, time, math

def run( cmd, log ):
    before = time.time();
    #time.sleep(0.2)
    #p = subprocess.check_call(cmd)
    total = round(time.time()-before,2)
    print("cmd = '%s'" % " ".join(cmd))
    print("runtime = %s s\n\n" % total, file=log)

if __name__ == '__main__':
    parser = optparse.OptionParser(usage="run_test.py <input images>")
    parser.add_option('--hmin', dest='hmin', default=-100, type='int')
    parser.add_option('--hmax', dest='hmax', default=100, type='int')
    parser.add_option('--vmin', dest='vmin', default=-20, type='int')
    parser.add_option('--vmax', dest='vmax', default=20, type='int')

    global opt
    (opt, args) = parser.parse_args()

    if not args or not len(args) == 2:
        print('\nERROR: Missing 2 input files')
        sys.exit(2)

    log = open("run_test_%s.log" % args[0].split('/')[-1], 'w')
    print("input1 = %s" % args[0], file=log)
    print("input2 = %s\n" % args[1], file=log)
    print("search range = %s %s %s %s\n" % (opt.hmin,opt.hmax,opt.vmin,opt.vmax), file=log)

    kernel_size_list = [15];
    search_exp_list = [1];
    #search_exp_list = [0.01,0.1,0.25,0.5,1,2,4,8];

    for kernel_size in kernel_size_list:
        for s_exp in search_exp_list:
            hc = (opt.hmax + opt.hmin)/2
            vc = (opt.vmax + opt.vmin)/2
            hw = (opt.hmax - opt.hmin) * math.sqrt(s_exp)/2.0
            if ( hw < 1 ):
                hw = 1
            vw = (opt.vmax - opt.vmin) * math.sqrt(s_exp)/2.0
            if ( vw < 1 ):
                vw = 1
            lhmin = int(round(hc - hw))
            lhmax = int(round(hc + hw))
            lvmin = int(round(vc - vw))
            lvmax = int(round(vc + vw))

            stereo_args = []
            stereo_args.extend(['--h-corr-min',str(lhmin)])
            stereo_args.extend(['--h-corr-max',str(lhmax)])
            stereo_args.extend(['--v-corr-min',str(lvmin)])
            stereo_args.extend(['--v-corr-max',str(lvmax)])
            stereo_args.extend(['--h-kernel',str(kernel_size)])
            stereo_args.extend(['--v-kernel',str(kernel_size)])
            stereo_args.extend(['--subpixel-kernel',str(kernel_size),str(kernel_size)])
            stereo_args.extend(['--xcorr-threshold','1'])
            stereo_args.extend(['--corr-seed-mode','0'])
            stereo_args.extend(['--prefilter-mode','0']) # Questionable?
            stereo_args.extend(args)
            stereo_args.extend(['/Users/zmoratto/projects/StereoPipeline/data/K10/black_left.tsai','/Users/zmoratto/projects/StereoPipeline/data/K10/black_right.tsai'])
            stereo_args.extend(['asp_%s_%s/asp_%s_%s' % (s_exp,kernel_size,s_exp,kernel_size)])

            patchm_args = []
            patchm_args.extend(['--h-corr-min',str(lhmin)])
            patchm_args.extend(['--h-corr-max',str(lhmax)])
            patchm_args.extend(['--v-corr-min',str(lvmin)])
            patchm_args.extend(['--v-corr-max',str(lvmax)])
            patchm_args.extend(['--xkernel',str(kernel_size)])
            patchm_args.extend(['--ykernel',str(kernel_size)])
            patchm_args.extend(['--tag',"pm_%s_%s" % (s_exp,kernel_size)])
            patchm_args.extend(args)

            print("Running search:%s kernel:%s" % (s_exp,kernel_size));
            run( ['patch_match'] + patchm_args, log )
            run( ['stereo_pprc'] + stereo_args, log )
            run( ['stereo_corr'] + stereo_args, log )
            run( ['stereo_rfne'] + stereo_args, log )

