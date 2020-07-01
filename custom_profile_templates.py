def base_template(args):
    args.use_gpu = True
    args.enable_profiling = True
    args.inference_only = True


def warmup(args):
    template1(args)
    args.template = 'warmup'

def template1(args):
    base_template(args)
    args.template = 'temp1'
    args.mini_batch_size = 16
    args.data_size = 160
    args.arch_sparse_feature_size = 16
    args.arch_embedding_size = '128-64-32'
    args.arch_mlp_bot = '32-16'
    args.arch_mlp_top = '4-2-1'


def template1_1(args):
    template1(args)
    args.template = 'd0=32'

def template1_2(args):
    template1(args)
    args.arch_sparse_feature_size = 32
    args.arch_mlp_bot = '64-32'
    args.template = 'd0=64'

def template1_2_MD(args):
    template1_2(args)
    args.template = 'd0=64_md'
    args.md_flag = True

def template1_2_QR(args):
    template1_2(args)
    args.template = 'd0=64_qr'
    args.qr_flag = True

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

# TEMPLATES = [warmup, template1, template2, template3, template4]
TEMPLATES = [warmup, template1_1, template1_2, template1_2_MD, template1_2_QR]
