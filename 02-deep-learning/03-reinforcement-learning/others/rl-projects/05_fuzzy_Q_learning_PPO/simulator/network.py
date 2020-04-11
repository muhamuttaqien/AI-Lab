from copy import deepcopy

import numpy as np
import pandas as pd

from .config import ConfigNet
from .make_net import GridNet

pd.set_option('display.precision', 4)


def get_free_id(df):
    
    return np.int64(0) if len(df) == 0 else df.index.values.max() + 1

class Network(object):
    
    net = GridNet(ConfigNet)
    component = []
    
    for s in net:
        
        component.append(s)
        if isinstance(net[s], list):
            net[s] = pd.DataFrame(np.zeros(0, dtype=net[s]), index=pd.Int64Index([]))
            
    def __init__(self, new=True):
        
        if new:
            self.net.clear()
            self.net = GridNet(ConfigNet)
            for s in self.net:
                if isinstance(self.net[s], list):
                    self.net[s] = pd.DataFrame(np.zeros(0, dtype=self.net[s]), index=pd.Int64Index([]))
        
        self.load = self.net.load
        self.generator = self.net.generator
        self.reset_results()
        
    def create_load(self, amount, unit, index=None, name=None):
        
        if index is None:
            index = get_free_id(self.net['load'])
            
        if index in self.net['load'].index:
            raise UserWarning('A load with the id %s already exists' % index)
            
        if name == None:
            name = 'Hour-' + str(index)
            
        self.net.load.loc[index, ['name', 'amount', 'unit']] = [name, amount, unit]
        
        return index
    
    def create_fuzzy(self, right, left, right0, left0, span, minup, mindown, index=None, name=None):
        
        if index is None:
            index = get_free_id(self.net['fuzzy'])
            
        if index in self.net['fuzzy'].index:
            raise UserWarning('A load with the id %s already exists' % index)
            
        if name == None:
            name = 'gen-' + str(index)
            
        self.net.fuzzy.loc[index, ['name', 'right', 'left', 'right0', 'left0', 'span', 'minup', 'mindown']] = [name, right, left, right0, left0, span, minup, mindown]
        
        return index
    
    def create_generator(self, u_max, u_min, ag, bg, cg, t_up, t_down, hs_up, cs_up, t_cold, t_initial, index=None, name=None, unit='[MW]'):
        
        if index is None:
            index = get_free_id(self.net['generator'])
            
        if index in self.net['generator'].index:
            raise UserWarning('A generator with the id %s already exists' % index)
            
        if name == None:
            name = 'gen-' + str(index)
            
        self.net.generator.loc[index, ['name', 'unit', 'u_max', 'u_min', 'a_g', 'b_g', 'c_g', 't_up', 't_down', 'hs_up', 'cs_up', 't_cold', 't_initial']] = [name, unit, u_max, u_min, ag, bg, cg, t_up, t_down, hs_up, cs_up, t_cold, t_initial]

        return index
    
    def all_element(self):
        
        for key in self.__dict__:
            print(key)
            
    def get_elements_to_empty(self):
        
        return ['cost', 'startup', 'on_off_schedule', 'spinnin_reserve', 'generation', 'priority', 'up', 'down', 'fuzzy', 'use_schedule', 'reward']
    
    def get_result_tables(self, element, suffix=None):
        
        res_empty_element = '_empty_res_' + element
        res_element = 'res_' + element
        if suffix is not None:
            res_element += suffix
        
        return res_element, res_empty_element
    
    def empty_res_element(self, element, suffix=None):
        
        res_element, res_empty_element = self.get_result_tables(element, suffix)
        self.net[res_element] = self.net[res_empty_element].copy()
        
    def init_element(self, element, suffix=None):
        
        res_element, res_empty_element = self.get_result_tables(element, suffix)
        index = self.net[element].index
        
        if len(index):
            res_columns = self.net[res_empty_element].columns
            self.net[res_element] = pd.DataFrame(np.nan, index=index, columns=res_columns, dtype='float')
        else:
            self.empty_res_element(element, suffix)
            
    def set_res(self):
        
        self.net['res_cost']['name'] = self.net['load']['name']
        self.net['res_cost'] = self.net['res_cost'].set_index('name')
        
        self.net['res_reward']['name'] = self.net['load']['name']
        self.net['res_reward'] = self.net['res_reward'].set_index('name')
        
        self.net['res_use_schedule']['name'] = self.net['load']['name']
        self.net['res_use_schedule'] = self.net['res_use_schedule'].set_index('name')
        
        self.net['res_fuzzy']['name'] = self.net['generator']['name']
        self.net['res_fuzzy'] = self.net['res_fuzzy'].set_index('name')
        
        self.net['res_up']['name'] = self.net['load']['name']
        self.net['res_up'] = self.net['res_up'].set_index('name')
        
        self.net['res_down']['name'] = self.net['load']['name']
        self.net['res_down'] = self.net['res_down'].set_index('name')
        
        self.net['res_priority']['name'] = self.net['load']['name']
        self.net['res_priority'] = self.net['res_priority'].set_index('name')
        
        self.net['res_startup']['name'] = self.net['load']['name']
        self.net['res_startup'] = self.net['res_startup'].set_index('name')
        
        self.net['res_on_off_schedule']['name'] = self.net['load']['name']
        self.net['res_on_off_schedule'] = self.net['res_on_off_schedule'].set_index('name')
        
        self.net['res_spinnin_reserve']['name'] = self.net['load']['name']
        self.net['res_spinnin_reserve'] = self.net['res_spinnin_reserve'].set_index('name')
        
        self.net['res_generation']['name'] = self.net['load']['name']
        self.net['res_generation'] = self.net['res_generation'].set_index('name')
        
    def reset_results(self, suffix=None):
        
        elements_to_empty = self.get_elements_to_empty()
        
        for element in elements_to_empty:
            self.empty_res_element(element, suffix)
        
        self.set_res()
        self.set_res_zero()
        
    def set_res_zero(self):
        
        for i in range(self.net._len):
            
            name = 'Hour-' + str(i)
            self.net.res_cost.loc[name,:] = 0
            self.net.res_startup.loc[name,:] = 0
            self.net.res_on_off_schedule.loc[name,:] = 0
            self.net.res_spinnin_reserve.loc[name,:] = 0
            self.net.res_generation.loc[name,:] = 0
            self.net.res_priority.loc[name,:] = 0
            self.net.res_up.loc[name,:] = 0
            self.net.res_down.loc[name,:] = 0
            self.net.res_use_schedule.loc[name,:] = 0
            self.net.res_reward.loc[name,:] = 0
            
        self.net.res_load = self.net.load
        self.net.res_generator = self.net.generator
        
        for j in range(self.net.gen_len):
            
            name = 'gen-' + str(j)
            u_name = name + 'fuzzy'
            self.net.res_fuzzy.at[name, 'fuzzy'] = u_name
            