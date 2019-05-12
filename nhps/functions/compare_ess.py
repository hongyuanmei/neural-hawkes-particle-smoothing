import os
import pickle
import argparse
import numpy as np

from nhps.sigtest.pair_perm import PairPerm


def main():
    parser = argparse.ArgumentParser("Give me two particle cache and I can do a significance test"
                                     "for their effective sampling size.")
    parser.add_argument(
        '-p', '--Path', nargs=2, type=str,
        help='The paths of these to cache'
    )

    args = vars(parser.parse_args())

    path_a, path_b = args['Path']
    if not os.path.exists(path_a):
        print('The first path you provided is not valid!')
        exit(1)
    if not os.path.exists(path_b):
        print('The second path you provided is not valid!')
        exit(1)

    weights_a, weights_b = map(lambda path_: pickle.load(open(path_, 'rb'))[0], [path_a, path_b])
    if not weights_a.shape == weights_b.shape:
        print('Shape not matched!')
        exit(2)

    ess_a, ess_b = map(lambda w_: w_.sum(axis=1) ** 2 / (w_ ** 2).sum(axis=1), [weights_a, weights_b])
    print('For the first result, ESS = {:.3f}'.format(np.mean(ess_a)))
    print('For the second result, ESS = {:.3f}'.format(np.mean(ess_b)))
    print('Significance test:')
    print('{:.8f}'.format(PairPerm().run(ess_a, ess_b)))
    print('For {}/{} of the particles, the first result performs better.'
          .format((ess_a > ess_b).sum(), len(ess_a)))
    print('For {}/{} of the particles, it\'s a tie.'
          .format((ess_a == ess_b).sum(), len(ess_a)))


if __name__ == '__main__':
    main()
