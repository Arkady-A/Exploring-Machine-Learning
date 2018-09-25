#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 18:43:38 2018

@author: ark
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as lines

# preparing axes
def prepareax(ax, xes=[-7,7], yes=[-7,7]):
    '''
    Changes plot to euclidean view
    '''
    x_min = xes[0]
    x_max = xes[1]
    y_min = yes[0]
    y_max = yes[1]
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xticks(range(x_min,x_max+1,1))
    ax.set_yticks(range(y_min,y_max+1,1))
    ax.set_aspect('equal')
    ax.grid(alpha = 0.1)

#**kargs - bad decision, but i'm too lazy to refactor
def draw_arrows(ax, origin_points, magnitudes, label, text_size=13, **kargs):
    '''
    Draws 2 arrows representing current basis
    ax - axes
    origin_points - start point of arrows, 2x2 numpy array [[x,x],[y,y]]
    magnitudes - directions from origin_poin, 2x2 numpy array [[x,x],[y,y]]
    label - label for arrows
    
    '''
    ax.arrow(*origin_points[:,0], *magnitudes[:,0], head_width=0.30*kargs['w_coef'], label=label,
             head_length=0.25*kargs['w_coef'], fc=kargs['color'], ec=kargs['color'], zorder=kargs['zorder'], 
             alpha=kargs['alpha'], length_includes_head=True)
    ax.arrow(*origin_points[:,1], *magnitudes[:,1], head_width=0.30*kargs['w_coef'], 
             head_length=0.25*kargs['w_coef'], fc=kargs['color'], ec=kargs['color'], zorder=kargs['zorder'], 
             alpha=kargs['alpha'], length_includes_head=True)
    if kargs['mark_text']==True:
        ax.text(magnitudes[0,1],magnitudes[1,1]+0.2,'j', zorder = 100, fontsize=text_size)
        ax.text(magnitudes[0,0],magnitudes[1,0]+0.2,'i', zorder = 100, fontsize=text_size)


def draw_line_points_change(ax, points1,points2, color_p, alpha=0.15,
                            linewidth=1,zorder=-1,linestyle='--'):
    '''
    Draws a line between points1 and points2
    '''
    for indx, point in enumerate(points1):
        ax.add_line(lines.Line2D
                    ([point[0],points2[indx,0]],[point[1],points2[indx,1]],
                     color = color_p['tl'], alpha=alpha,linewidth=linewidth,
                            linestyle=linestyle, zorder=1
                            )
                    )
                    
def draw_transformed_axis(ax, transformation,text_size=13, names=['n_x','n_y'], 
                          gridsize=(-5,5),color='black',alpha=1.0,
                          linestyle='-',linewidth=1, zorder=10):
    names=names.__iter__()
    for axes in transformation:
        ax.text(axes[0]*gridsize[1],axes[1]*gridsize[1],next(names),
                fontsize=text_size)
        ax.add_line(lines.Line2D([axes[0]*gridsize[0],axes[0]*gridsize[1]],
                                  [axes[1]*gridsize[0],axes[1]*gridsize[1]],
                                  color= color,alpha=alpha,linestyle=linestyle,
                                  linewidth=linewidth,zorder=zorder))
         
def draw_transformed_grid(ax, transformation, gridsize=(-5,5), 
                          color='grey',alpha=0.2,
                          linestyle='-',linewidth=1, zorder=1):
    for j in range(0,2):
        mover_x = transformation[0,1-j]*gridsize[1]
        mover_y = transformation[1,1-j]*gridsize[1]
        for i in range(gridsize[0],gridsize[1]+1):
             ax.add_line(
                    lines.Line2D(
                            [i*transformation[0,0+j]-mover_x,
                             i*transformation[0,0+j]+mover_x],
                            [i*transformation[1,0+j]-mover_y,
                             i*transformation[1,0+j]+mover_y],
                            color = color, alpha=alpha,linewidth=linewidth,
                            linestyle=linestyle, zorder=1
                            ))   


def visualize_transfromation(ax, s_t, a_t, name, color_p):
    """
    Visualize matrix transformation
    Keyword arguments:
    ax - matplotlib axes
    s_t - start transformation
    a_t - next transformation
    name - title name for ax
    color_p = color pallete
    
    Returns: 
    n_t  - new transfromation
    """
    ax.set_title(name)
    prepareax(ax)
    n_t = np.matmul(s_t, a_t)
    #print(n_t
    
    draw_arrows(ax, np.array([[0,0],[0,0]]), s_t, 'Before transformation', color=color_p['bt'],
                                 zorder=20,alpha=1,w_coef=1,mark_text=False)
    draw_arrows(ax, np.array([[0,0],[0,0]]), n_t, 'After transfomration',color=color_p['at'],
                                 zorder=21,alpha=1,w_coef=1,mark_text=True)
    draw_arrows(ax, s_t, n_t-s_t, 'Change', color=color_p['tl'],
                    zorder=4,alpha=0.2,w_coef=0,mark_text=False)                                 
        
    return n_t