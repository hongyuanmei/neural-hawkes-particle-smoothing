# -*- coding: utf-8 -*-
"""

Drawers

@author: hongyuan
"""

import time
import numpy
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#@profile
class Drawer(object):

    def __init__(self, with_sub_points=True):
        self.with_sub_points = with_sub_points

    def setResults(self, path_figure, results):

        self.pfigure = path_figure
        #self.results = { 'MAP': {}, 'MBR': {} }
        self.results = {'MBR': {}}

        for decode_method in self.results.keys():

            self.results[decode_method]['nhpf'] = {}
            for data in results[decode_method]['nhpf']:
                self.results[decode_method]['nhpf'][data] = results[decode_method]['nhpf'][data]

            self.results[decode_method]['nhps_inter'] = {}
            for modelid in results[decode_method]['nhps']:
                self.results[decode_method]['nhps_inter'][modelid] = {}
                for data in results[decode_method]['nhps'][modelid]:
                    self.results[decode_method]['nhps_inter'][modelid][data] = results[decode_method]['nhps'][modelid][data]

            self.list_id = sorted(self.results[decode_method]['nhps_inter'].keys())
            self.id_max = self.list_id[-1]
            self.results[decode_method]['nhps'] = self.results[decode_method]['nhps_inter'][self.id_max]


    def draw(self, name_figure='no_name',
            x_min=None, x_max=None, y_min=None, y_max=None):

        self.draw_pareto(
            xaxis='insdel_per_true', yaxis='total_transport_per_true',
            name_figure=name_figure+'_x-insdelpertrue_y-totaltransportpertrue',
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


    def draw_pareto(self, xaxis, yaxis, name_figure='no_name',
                    x_min=None, x_max=None, y_min=None, y_max=None):

        # draw Pareto
        fig, ax1 = plt.subplots()
        colors = {
            'MAP': {'nhpf': 'g', 'nhps': 'm'},
            'MBR': {'nhpf': 'b', 'nhps': 'r'}
        }

        labels = {
            'MAP': {'nhpf': 'particle filtering', 'nhps': 'particle smoothing'},
            'MBR': {'nhpf': 'particle filtering', 'nhps': 'particle smoothing'}
        }

        markersize = mpl.rcParams['lines.markersize'] # according to reading their code
        # ratio = 1.0 - markersize / numpy.sqrt(directionx**2 + directiony**2)
        # to avoid transforming between point coordinate and data coordinate
        # we assume (reasonably) that each dot's radius is ~1/50=0.02
        x_max = x_max or max(
            self.results['MBR']['nhpf'][xaxis].max(), self.results['MBR']['nhps'][xaxis].max()
        )
        x_min = x_min or min(
            self.results['MBR']['nhpf'][xaxis].min(), self.results['MBR']['nhps'][xaxis].min()
        )
        y_max = y_max or max(
            self.results['MBR']['nhpf'][yaxis].max(), self.results['MBR']['nhps'][yaxis].max()
        )
        y_min = y_min or min(
            self.results['MBR']['nhpf'][yaxis].min(), self.results['MBR']['nhps'][yaxis].min()
        )

        x_gap = x_max - x_min
        y_gap = y_max - y_min

        x_margin = 0.2 * x_gap
        y_margin = 0.2 * y_gap

        # set lim so we have fixed pixels for each x and y
        # so when we draw arrow lines of the same length
        # we can safely use length in pixel to compute data coord
        ax1.set_xlim(x_min - 0.05 * x_gap, x_max + x_margin)
        ax1.set_ylim(y_min - 0.05 * y_gap, y_max + y_margin)

        temp = ax1.transData.transform([(0.0, 0.0), (1.0, 0.0)])[:,0]
        pixel_per_x = temp[1] - temp[0]

        temp = ax1.transData.transform([(0.0, 0.0), (0.0, 1.0)])[:,1]
        pixel_per_y = temp[1] - temp[0]

        weight_min = 0.1 * x_gap
        height_min = 0.1 * y_gap
        radius = x_max - x_min
        radius *= 0.01
        sizes = {
            'x_max': x_max, 'x_min': x_min, 'x_gap': x_gap,
            'y_max': y_max, 'y_min': y_min, 'y_gap': y_gap,
            'weight_min': weight_min, 'height_min': height_min,
            'radius': radius,
            'pixel_per_x': pixel_per_x, 'pixel_per_y': pixel_per_y,
            'x_margin': x_margin, 'y_margin': y_margin
        }

        self.draw_pareto_one_decode_method(
            fig, ax1, 'MBR', colors, labels, sizes, xaxis, yaxis)

        fig.tight_layout()

        fig.legend(loc='upper right', bbox_to_anchor=[0.9, 0.9])

        plt.xlabel('(# insertion + # deletion)/# true')
        plt.ylabel('total_align_cost/# true')

        path_to_save = os.path.join(
            self.pfigure, 'pareto_{}.pdf'.format(name_figure))
        fig.savefig(path_to_save, bbox_inches='tight')

    def draw_pareto_one_decode_method(
        self, fig, ax1, decode_method, colors, labels, sizes, xaxis, yaxis):

        ax1.plot(
            self.results[decode_method]['nhpf'][xaxis], self.results[decode_method]['nhpf'][yaxis],
            colors[decode_method]['nhpf'] + 'o',
            label=labels[decode_method]['nhpf']
        )
        ax1.plot(
            self.results[decode_method]['nhps'][xaxis], self.results[decode_method]['nhps'][yaxis],
            colors[decode_method]['nhps'] + '^', label=labels[decode_method]['nhps']
        )

        list_xy = []

        for i in range(self.results[decode_method]['nhpf']['costs'].shape[0]):

            # draw red arrows from PF to PS
            startx = self.results[decode_method]['nhpf'][xaxis][i]
            starty = self.results[decode_method]['nhpf'][yaxis][i]
            endx = self.results[decode_method]['nhps'][xaxis][i]
            endy = self.results[decode_method]['nhps'][yaxis][i]

            # we need to do some simple tedious math to make the figure pretty
            directionx = endx - startx
            directiony = endy - starty

            red_props = dict(
                arrowstyle='->',
                ls='--',
                shrinkA=4,
                shrinkB=3,
                linewidth=1,
                color='red'
            )
            ax1.annotate('', xytext=[startx, starty], xy=[endx, endy],
                         arrowprops=red_props, size=20)

            list_xy.append( ( startx, starty ) )

        avg_len_in_pixel = min(
            sizes['x_margin'] * sizes['pixel_per_x'],
            sizes['y_margin'] * sizes['pixel_per_y']
        )

        for i, cost_value in enumerate(self.results[decode_method]['nhpf']['costs']):

            endx, endy = list_xy[i]
            # draw blue arrows to PF

            slope = self.results[decode_method]['nhpf']['costs'][i]

            height_in_pixel = avg_len_in_pixel * slope / (numpy.sqrt(1.0 + slope**2))
            weight_in_pixel = height_in_pixel / slope

            height = height_in_pixel / sizes['pixel_per_y']
            weight = weight_in_pixel / sizes['pixel_per_x']

            startx = endx + 1.0 * weight
            starty = endy + 1.0 * height

            directionx = endx - startx
            directiony = endy - starty

            ratio = 1.0 - sizes['radius'] / numpy.sqrt(directionx**2 + directiony**2)

            if ratio > 0.0:
                directionx *= ratio
                directiony *= ratio

            blue_props = dict(
                arrowstyle='->',
                shrinkA=0,
                shrinkB=3,
                linewidth=1,
                color='blue'
            )
            ax1.annotate(
                '', xy=[endx, endy], xytext=[startx, starty],
                arrowprops=blue_props, size=20
            )

            loc_x, loc_y = startx, starty

            def adjust(lim, loc_1, loc_2, step_1, step_2):
                if loc_2 + step_2 > lim[1]:
                    loc_1 += step_1
                    loc_2 -= step_2
                return loc_1, loc_2

            xlim, ylim = ax1.get_xlim(), ax1.get_ylim()
            step_x = abs(xlim[0] - xlim[1]) / 12
            step_y = abs(ylim[0] - ylim[1]) / 30
            loc_x, loc_y = adjust(ylim, loc_x, loc_y, step_x/5, step_y)
            loc_y, loc_x = adjust(xlim, loc_y, loc_x, step_y/2, step_x)

            label_loc = [loc_x, loc_y]

            ax1.annotate(
                'C={:.2f}'.format(cost_value), xy=label_loc, xytext=label_loc,
                color='blue'
            )

        r"""
        plot the other red dots (use alpha and smaller size)
        """
        for modelid in self.list_id:
            ax1.plot(
                self.results[decode_method]['nhps_inter'][modelid][xaxis],
                self.results[decode_method]['nhps_inter'][modelid][yaxis],
                colors[decode_method]['nhps'] + 'o',
                alpha=0.2, markersize=3, markeredgewidth=0.0
            )
