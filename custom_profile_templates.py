def base_template(args):
    args.use_gpu = True
    args.enable_profiling = True


def warmup(args):
    base_template(args)
    args.template = 'warmup'
    args.mini_batch_size = 16
    args.data_size = 160

def template1(args):
    base_template(args)
    args.template = 'temp1'
    args.mini_batch_size = 16
    args.data_size = 160


def template2(args):
    base_template(args)
    args.template = 'temp2'
    args.mini_batch_size = 32
    args.data_size = 320

def template3(args):
    base_template(args)
    args.template = 'temp3'
    args.mini_batch_size = 64
    args.data_size = 640

def template4(args):
    base_template(args)
    args.template = 'temp4'
    args.mini_batch_size = 128
    args.data_size = 1280

TEMPLATES = [warmup, template1, template2, template3, template4]
