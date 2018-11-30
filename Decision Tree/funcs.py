# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:36:35 2018

@author: ark4d
"""
import numpy as np


class Probability():
    '''
    This class contains methods to work with probability
    '''     
    @staticmethod
    def get_distr(data, un_values=[]):
        '''
        Creates array with probability of each values that to occur
        Parameters 
        ----------
        data : numpy-array(int)
            data probability of element which will be tested
        un_values : array-like(int), optional
            unique values probability of which will be created
        Returns
        ----------
        numpy-array(float)
            chace of occurence
        '''
        if len(un_values)==0:
            un_values=np.unique(data)
        values, count = [],[] 
        for value in un_values:
            values.append(value)
            count.append(len(data[data==value]))
        dist = np.array([values, np.array(count)/data.shape[0]])
        return dist
    

    @staticmethod
    def dist_table(data1, data2):
        '''
        Create table that represent joint probability of two distributions
        
        Parameters
        ----------
        data1 : numpy-array (int)
            data 
        data2 : numpy-array (int)
            data
            
        Returns
        ----------
        list 
            unique values of data1
        list 
            unique values of data2
        numpy matrix
            joint probability of two distribution 
        
        Raises
        ----------
        ValueError
            if `data1` shape is not equal to `data2` shape
        '''
        if data1.shape == data2.shape:
            size = data1.shape[0]
            un_x_values = np.unique(data1)
            un_y_values = np.unique(data2)
            table = np.zeros((un_x_values.shape[0], un_y_values.shape[0]))
            for row in np.transpose(np.array([data1,data2])): 
                table[
                    np.where(un_x_values==row[0]),
                    np.where(un_y_values==row[1])
                ]+=1
            return un_x_values, un_y_values, table/size
        else:
            raise ValueError('{} != {}'.format(data1.shape, data2.shape))
        
    
class Itheory():
    @staticmethod
#    note: if you were to check this function with the function in scipy.stats 
#    you will get different values# because scipy calculates entropy with base 
#    e instead of 2. The values is different, but they showing same thing. you 
#    can say that the function in scipy return decibels of entropy whereas
#    the function beneath returns bits (shannons)
#    of entropy.
    def entropy(pmf):
        '''
        Calculates entropy
        
        Parameters
        ----------
        pmf : numpy-array (float)
            Probabilities of each element to occur
        
        Retuns
        ----------
        float
            entropy
        '''
        entrp = -(pmf*np.nan_to_num(np.log2(pmf),0)).sum()
        return entrp
    
    @staticmethod
    def joint_entropy(table):
        '''
        Calculates joint entropy of 2 distributions
        
        Parameters
        ----------
        table : numpy matrix
            Joint probability distribution matrix (can be generated
            by `dist_table` function)
            
        Returns
        ----------
        float
            joint entropy 
        '''
        j_ent = np.nan_to_num(table*np.log2(table),0).sum()
        return -j_ent
    
    @staticmethod
    def conditional_entropy(table, axis_marginal):
        '''
        Calculates conditional entropy of 2 distributions
        
        Parameters
        ----------
        table : numpy matrix
            Joint probability distribution matrix (can be generated
            by `dist_table` function)
        
        axis_marginal : int, 1 or 0
            Determines given which data entropy will be calculater
            0 - H(`data1`|`data2`)
            1 - H(`data2`|`data1`)
            look in `dist_table` for definition of `data1` and `data2`
        Returns
        ----------
        float
            conditional entropy
        '''
        #informal: relatively hard function to comprehend.
        marg = np.apply_along_axis(lambda x: x.sum(),axis_marginal,table)
        table=np.apply_along_axis(lambda x: x*marg**-1,1-axis_marginal,table)
        table=np.apply_along_axis(lambda x: Itheory.entropy(x),axis_marginal,table)
        table = np.nan_to_num(table,0)
        c_ent = np.matmul(marg, table)
        return c_ent
    
    @staticmethod
    def rel_entropy(prob_dist1, prob_dist2):
        '''
        Calculates relative entropy
        
        Parameters
        ----------
        prob_dist1 : numpy-array
            marginal probability of a distribution
        prob_dist2 : numpy-array
            marginal probability of a distribution
            
        Returns
        ----------
        float
            relative entropy
        '''
        div = prob_dist1/prob_dist2
        if np.isinf(div).any():
            return np.inf
        else:
            log = np.log2(div)
            log[np.isinf(log)]=0
            log[np.isnan(log)]=0
            rel_entropy = np.sum(prob_dist1*log)
            return rel_entropy
        
    @staticmethod
    def mutual_information(prob_dist1, prob_dist2, table):
        '''
        Calculates mutual information between 2 distributions
        
        Parameters
        ----------
        prob_dist1 : numpy-array
            marginal probability of a distribution
        prob_dist2 : numpy-array
            marginal probability of a distribtuion
        table: numpy-matrix
            Joint probability distribution matrix (can be generated
            by `dist_table` function)
            
        Returns
        ----------
        float
            Mutual information
        '''
        div = table/(prob_dist1.reshape(-1,1)* prob_dist2)
        if np.isinf(div).any():
            return np.inf
        else:
            log = np.log2(div)
            log[np.isinf(log)]=0
            log[np.isnan(log)]=0
            mutual_inf = np.sum(table*log)
            return mutual_inf

# TO-DO: Add docs to utility.
class Utility():
    
    @staticmethod
    def make_ax_look_good(ax):
        '''
        Makes axes look fancy :)
        
        Parameters
        ----------
        ax : matplotlib axes
        '''
        ax.grid(alpha=0.2)
        for spine in ax.spines:
                ax.spines[spine].set_visible(False)
                
    @staticmethod
    def less_ent_d(s, step_m = 2, low_boundary = 4, times =2 ):
        '''
        Changes distribution such that the entropy descends.
        '''
        mult = 2 
        s_m = s.mean().round().astype('int')
        vals = np.array(np.unique(s,return_counts=True))
        vals_gtlw = (vals[0,vals[1,:]>low_boundary])
        vals_gtlw = vals_gtlw[(vals_gtlw<s_m) | (vals_gtlw>s_m)]
        np.set_printoptions(threshold=np.nan)
        for i in range(0,times):
            for indx, val in enumerate(vals_gtlw[vals_gtlw<s_m][::-1]):
                step_arr=np.zeros(shape=(len(s[s==val]),),dtype='int')
                step_arr[:int(len(s[s==val])/3)]=+step_m
                s[s==val]+=step_arr
                mult+=1
            mult=2
            for indx, val in enumerate(vals_gtlw[vals_gtlw>s_m]):
                step_arr=np.zeros(shape=(len(s[s==val]),),dtype='int')
                step_arr[:int(len(s[s==val])/3)]=-step_m
                s[s==val]+=step_arr
                mult+=1
            
        return s
   
    @staticmethod
    def less_ent(dist, multiplier=0.01,steps=2, low_border=0.004):
        '''
        Chages data such that the entropy descends.
        '''
        size = len(dist)
        s_indx = np.argmax(dist!=0)
        e_indx = np.argmax(np.flip(dist)!=0)
        m_indx_d = np.round(((size-(s_indx+e_indx))/2)).astype('int')
        m_indx = m_indx_d+s_indx
#        print(s_indx, -e_indx, m_indx)
        for step in range(0,steps):
            for i in range(s_indx,m_indx):
                distance = np.abs(m_indx - i)
                distance_coef = distance/m_indx_d
                buff = dist[i]*distance_coef* multiplier
                if (dist[i]-buff)<(low_border) and dist[i]>=low_border:
                    buff = (dist[i]-low_border)
                dist[i], dist[i+1] = dist[i]-buff, dist[i+1]+buff
                
            for i in range(size-1-e_indx, m_indx,-1):
                distance = np.abs(m_indx - i)
                distance_coef = distance/m_indx_d
                buff = dist[i]*distance_coef* multiplier
                if (dist[i]-buff)<(low_border) and dist[i]>=low_border:
                    buff = (dist[i]-low_border)
                dist[i], dist[i-1] = dist[i]-buff, dist[i-1]+buff
        return dist
    
    #this function shuffles values of !probability distribution!
    #dist: a probability distribution
    #times: greater - more shufflier 
    #multiplier: greater - more shufflier
    @staticmethod
    def shuffle_dist(dist, times=10, multiplier=0.4):
        '''
        Changes values of distribtuions. Depends on variables that are passed.

        Parameters
        ----------
        dist : nparray
            distribution of a random variable
        times : int
            amount of times for the process to repeat
        multiplier : float
            determines how much the function will change the distribution on a iteration

        Returns
        ----------
        dist : nparray
            changed distribtuion
        '''
        deck = dist[dist!=0]
        start_index = np.argmax(dist!=0)
        end_index = start_index+deck.shape[0]
        buff=0
        for i in range(0, times):
            for i in range(start_index, end_index):
                if np.random.choice([True,False]):
                    buff, dist[i] = dist[i]*multiplier, (dist[i]-dist[i]*multiplier)+buff
            dist[np.random.randint(start_index, end_index)]+=buff
            buff=0
        return dist
    
    @staticmethod
    def _decide(chance):
        if np.random.rand()<chance:
            return True
        else:
            return False
        
    @staticmethod
    def shuffle_data(data, chance = 0.5, times=10, count_low_boundary=1, count_max_boundary=40):
        '''
        Changes data by changing a value to another value. Another value can be picked from range of
        unique values that are presented in the data. 
        '''
        uni,vals = np.unique(data,return_counts = True)
        for j in range(0, times):
            for i in range(data.shape[0]):
                if (vals[np.where(uni==data[i])]>count_low_boundary)and(vals[np.where(uni==data[i])]<count_max_boundary):
                    if Utility._decide(chance):
                        n_i = np.random.randint(low=0,high=data.shape[0])
                        vals[np.where(uni==data[i])]-=1
                        data[i]=data[n_i]
                        vals[np.where(uni==data[n_i])]+=1
                else:
                    pass
        return data
