import os,subprocess, sys, shutil

target = sys.argv[1]

pwd = os.getcwd()

sdfs = [x for x in os.listdir(target) if '.sdf' in x]

os.chdir(target)

for sdf in sdfs:
    conf_name = sdf.split('.')[0]
    try:
        with open(conf_name+'.log', 'w') as out:
            subprocess.call(['xtb', '{}.sdf'.format(conf_name), '-opt'], stdout=out, stderr=out)
    except:
        continue
    
    shutil.move('xtbopt.sdf', '{}_opt.sdf'.format(conf_name))
    
    try:
        with open(conf_name+'_freq.log', 'w') as out:
            subprocess.call(['xtb', '{}_opt.sdf'.format(conf_name), '-ohess'], stdout=out, stderr=out)
    except:
        continue
    
    os.remove('hessian')
    os.remove('vibspectrum')
    
    
os.chdir(pwd)
