from .gmot import build as build_gmot

def build_model(args):
    arch_catalog = {
        'gmot': build_gmot,
    }
    assert args.meta_arch in arch_catalog, 'invalid arch: {}'.format(args.meta_arch)
    build_func = arch_catalog[args.meta_arch]
    return build_func(args)
