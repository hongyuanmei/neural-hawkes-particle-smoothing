import pickle
import os
import argparse
from matplotlib import pyplot as plt
import numpy as np
from nhps.functions.draw_pareto import draw_pareto

from nhps.io.color_fig import draw_colored_fig


def parse():
    parser = argparse.ArgumentParser('Draw kernel density estimation figures.')
    parser.add_argument('-p', '--Path', required=True,
                        help='The path of the dataset.')
    parser.add_argument('-k', '--Keyword', type=str, nargs='+',
                        help='The keyword for dataset. E.g. [add, dim=256]')
    parser.add_argument('-sp', '--Split', required=True,
                        choices=['dev', 'test'],
                        help='Evaluate on dev or test dataset?')
    parser.add_argument('-m', '--Margin', type=float, default=0.3,
                        help='The ratio of margin to the whole figure.')
    parser.add_argument('-sg', '--Sigma', type=float, default=0.4,
                        help='Sigma for normal kernel. The larger sigma is, the sharper'
                        + 'kde figure is.')
    parser.add_argument('-ns', '--NStep', type=int, default=500,
                        help='The number of cells per side of mesh grid.')
    parser.add_argument('-dpi', '--DPI', type=int, default=500,
                        help='The DPI of the output PNG image.')
    parser.add_argument('-sn', '--SavedName', type=str,
                        help='The saved file name.')
    parser.add_argument('-vl', '--VerticalLine', action='store_true',
                        help='Whether to draw a vertical line.')
    parser.add_argument('-dp', '--DrawPareto', action='store_true',
                        help='Whether to draw pareto figs.')
    parser.add_argument('-dk', '--DrawKDE', action='store_true',
                        help='Whether to draw KDE figs')
    parser.add_argument('-xi', '--XMin', type=float, default=None,
                        help='The min x value in pareto figs')
    parser.add_argument('-xa', '--XMax', type=float, default=None,
                        help='The max x value in pareto figs')
    parser.add_argument('-yi', '--YMin', type=float, default=None,
                        help='The min y value in pareto figs')
    parser.add_argument('-ya', '--YMax', type=float, default=None,
                        help='The max y value in pareto figs')
    parser.add_argument('-ls', '--LabelSize', type=int, default=None,
                        help='size of x and y labels')
    parser.add_argument('-as', '--AnnotateSize', type=int, default=None,
                        help='size of xxx nats in figure')

    args = parser.parse_args()
    args = vars(args)
    if args['Keyword'] is None:
        args['Keyword'] = list()
    if args['SavedName'] is None:
        if len(args['Keyword']) == 0:
            args['SavedName'] = 'DefaultName'
        else:
            args['SavedName'] = '_'.join(args['Keyword'])
    return args


def main():
    args = parse()
    pts = list()
    centroids = list()
    for root, folders, _ in os.walk(args['Path']):
        for folder in folders:
            file_path = os.path.join(root, folder, '{}.results.pkl'.format(args['Split']))
            if not os.path.exists(file_path):
                continue
            if not all([word in file_path for word in args['Keyword']]):
                continue
            with open(file_path, 'rb') as fp:
                rst = pickle.load(fp)
            if args['DrawPareto']:
                draw_pareto(os.path.join(root, folder), args['Split'], x_min=args['XMin'],
                            x_max=args['XMax'], y_min=args['YMin'], y_max=args['YMax'])
            if not args['DrawKDE']:
                continue
            rst = rst['LogProposal']
            nhps_rst = rst['nhps']
            nhps_rst = nhps_rst[max(nhps_rst.keys())]
            nhpf_rst = rst['nhpf']
            pts_ = np.array([nhps_rst['all_avg_proposals'], nhpf_rst['all_avg_proposals']]).T
            pts.append(pts_)
            centroids.append([nhpf_rst['avg_proposal'], nhps_rst['avg_proposal']])
    if not args['DrawKDE']:
        return
    print('{} results found!'.format(len(pts)))
    centroids = np.array(centroids).T
    draw_colored_fig(
        pts, sigma=args['Sigma'], centroids=centroids,
        n_step=args['NStep'], margin=args['Margin'],
        label_size=args['LabelSize'],
        anno_size=args['AnnotateSize'] )
    print('Finished! Saving now.')
    save_path = os.path.join(args['Path'], args['SavedName'] + '.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=args['DPI'])
    # uncomment the following line for debugging
    # plt.show()


if __name__ == '__main__':
    main()
