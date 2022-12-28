import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)

    ## DIRECTORIES
    parser.add_argument('--fscd_dir', default='/home/intern/Desktop/datasets/FSCD147', type=str)
    parser.add_argument('--gmot_dir', default='/home/intern/Desktop/datasets/GMOT', type=str)
    parser.add_argument('--mot17_dir', default='/home/intern/Desktop/datasets/GMOT', type=str)

    parser.add_argument('--output_dir', default='./outputs/', type=str,     help="where to save logs/other") 

    parser.add_argument('--pretrained', default=None, type=str,             help="model's initializion weights  .pth")
    parser.add_argument('--resume', default=None, type=str,                 help="checkpoint to restart the training from")


    ## GENERAL
    parser.add_argument('--device', default='cuda',                         help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    # * Training settings
    parser.add_argument('--meta_arch', default='gmot', type=str,            help="model architecture to be used")
    parser.add_argument('--dataset_file', default=['e2e_fscd'], nargs='+',  help="datasets to use for training, will be joined")
    parser.add_argument('--epochs', default=30, type=int,                   help="fscd:250; mot17:50")
    parser.add_argument('--small_ds', action='store_true',                  help="use smaller portion of dataset")
    parser.add_argument('--debug', action='store_true',                     help="logs results & visualizations")

    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets',], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,         help='gradient clipping max norm')

    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', action='store_true')
    parser.add_argument('--sample_mode', type=str, default='random_interval')
    parser.add_argument('--sample_interval', type=int, default=20)
    parser.add_argument('--sampler_lengths', type=int, nargs='*', default=[5])

    # * Test
    parser.add_argument('--max_size', default=1333, type=int)
    parser.add_argument('--val_width', default=800, type=int)
    parser.add_argument('--t_dataset_file', default='e2e_fscd',             help="dataset or folder of images")


    ## ARCHITECTURE
    parser.add_argument('--embedd_dim', default=256, type=int,              help="Size of the embeddings (num channels)")
    
    # * PreBackbone
    parser.add_argument('--img_prep', default='padder', type=str,           help="padding, resizing, ...")

    # * Backbone
    parser.add_argument('--backbone', default='revbifpn', type=str,         help="name of the backbone to use")
    parser.add_argument('--num_feature_levels', default=4, type=int,        help='number of feature levels')

    # * Mixer
    parser.add_argument('--mix_return', default=['ch','q','feat'], type=str, nargs='+', help="what to return from mixer")
    parser.add_argument('--mix_arch', default='motr', type=str,         help="what to return from mixer")
    parser.add_argument('--q_extractor', default='circle', type=str,         help="how to go from exemplar feature maps to less queries (avg, multiple)")

    # * Position Embedding
    parser.add_argument('--position_embedding', default='sine_sum', type=str, choices=('sine_sum', 'sine_concat'),
                                                                            help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--use_fnn', action='store_true',                   help="pass encodings through a FFNN before and after the decoder")

    # * Transformer
    parser.add_argument('--exemp_features_key', action='store_true',        help="special version of DEF Transformer: queries learned; concat exemplar to keys")
    parser.add_argument('--num_queries', default=100, type=int,             help="Number of query slots") #NOTE: MOT20=300,   DANCETRACK=10, ....
    parser.add_argument('--dec_layers', default=6, type=int,                help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float,               help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=2, type=int,                    help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Variants of Deformable DETR
    # parser.add_argument('--with_box_refine', default=True, action='store_true')
    # parser.add_argument('--two_stage', default=False, action='store_true')
    # parser.add_argument('--accurate_ratio', default=False, action='store_true')

    # LOSS
    # * Matcher
    parser.add_argument('--mix_match', action='store_true',)
    parser.add_argument('--set_cost_class', default=2, type=float,          help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,           help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,           help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=4, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.12, type=float)

    return parser
