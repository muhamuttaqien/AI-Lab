import numpy as np
import pandas as pd

from simulator.network import Network
from simulator.excel_read_write import ExcelNet

from .fuzzy import Fuzzy
from .cost_calc import Cost_calc
from .cons_imp import Cons_imp


class Environment():
    
    def __init__(self):
        
        self.net = Network(new=True)
        
        ExcelNet(self.net)
        
        self.state = [0, 0, 0, 0, 0, 0]
        self.state_size = len(self.state)
        
        self.action = ['on', 'off']
        self.action_size = len(self.action)
        
        self.fuzzy = self.net.net.res_fuzzy
        self.cost_set = Cost_calc(self.net)
        self.cons_set = Cons_imp(self.net)
        
        self.reset()
        self.create_fuzzy()
    
    def reset(self):
        
        self.net.reset_results()
        gen = 'gen-0'
        _, gen_id = gen.split('-')
        gen_id = int(gen_id)
        self.hourly_setup(gen, 'Hour-0')
        val = self.net.net.res_generator.loc[:, 't_initial'].values > 0
        vals = val.astype(int)
        self.net.net.res_up.loc['Hour-0', :] = self.net.net.res_generator.loc[gen_id, 't_initial'].T * vals
        
    def create_fuzzy(self):
        
        fuzzy = self.net.net.fuzzy
        
        print(fuzzy)
        
        cost = self.net.net.res_cost
        mindown = self.net.net.res_generator
        h = 'Hour-0'
        
        for i in range(self.net.net.gen_len):
            
            generator = 'gen-' + str(i)
            self.fuzzy.at[generator, 'f_gen'] = Fuzzy([0, fuzzy.at[i, 'left0'], fuzzy.at[i, 'span']])
            self.fuzzy.at[generator, 'f_gen'].trapmf([fuzzy.at[i, 'right0'], fuzzy.at[i, 'right0'], fuzzy.at[i, 'right'], fuzzy.at[i, 'right']], 'g_low')
            self.fuzzy.at[generator, 'f_gen'].trapmf([fuzzy.at[i, 'right'], fuzzy.at[i, 'right'], fuzzy.at[i, 'left0'], fuzzy.at[i, 'left0']], 'g_high')
            
            self.fuzzy.at[generator, 'f_cost'] = Fuzzy([cost.loc[h, :].min()-50, cost.loc[h, :].max()+50, 50])
            self.fuzzy.at[generator, 'f_cost'].trinf([cost.loc[h, :].min(), cost.loc[h, :].max(), cost.loc[h, :].max()], 'cost')
            
            self.fuzzy.at[generator, 'f_minup'] = Fuzzy([0, fuzzy.at[i, 'minup']+10, 1])
            self.fuzzy.at[generator, 'f_minup'].trinf([0, fuzzy.at[i, 'minup'], fuzzy.at[i, 'minup']], 'minup')
            
            self.fuzzy.at[generator, 'f_mindown'] = Fuzzy([0, mindown.at[i, 't_down']+10, 1])
            self.fuzzy.at[generator, 'f_mindown'].trinf([0, mindown.at[i, 't_down'], mindown.at[i, 't_down']+10], 'mindown')

            self.fuzzy.at[generator, 'f_startup'] = Fuzzy([0, mindown.at[i, 'cs_up']+10, mindown.at[i, 'hs_up']])
            self.fuzzy.at[generator, 'f_startup'].trinf([0, mindown.at[i, 'cs_up'], mindown.at[i, 'cs_up']+10], 'startup')
    
    def hourly_setup(self, generator, hour):
        
        if generator == 'gen-0':
            self.cost_set.f_cost(hour)
            self.cons_set.priority_list(hour)
            self.cons_set.sefty_spining(hour)
            
        if generator == 'gen-9':
            self.cons_set.set_generation(hour)
            self.cons_set.minimum_up_down(hour)
            self.cost_set.stup_cost(hour)
    
    def step(self, generator, hour, action):
        
        self.hourly_setup(generator, hour)
        state = self.create_state(generator, hour, action)
        reward = self.create_reward(generator, hour, action)
        
        if generator == 'gen-9':
            reward = self.create_treward(generator, hour)

        return state, reward
    
    def create_state(self, generator, hour, action):
        
        state = []
        state.append(self.capacity(generator, hour))
        state.append(self.cost(generator, hour))
        state.append(self.minimum_up(generator, hour))
        state.append(self.minimum_down(generator, hour))
        state.append(self.startup(generator, hour))
        state.append(self.state_fc(generator, hour))
        
        return state
                        
    def capacity(self, generator, hour):
        
        _, hour_id = hour.split('-')
        hour_id = int(hour_id)
        
        _, gen_id = generator.split('-')
        gen_id = int(gen_id)
        
        t_demand = self.net.net.res_load.at[hour_id, 'amount']
        gen_umax = self.net.net.res_generator.loc[gen_id, 'u_max']
        gen = gen_umax/ t_demand
        
        return gen
        
    def cost(self, generator, hour):
        
        cost = self.net.net.res_cost.at[hour, generator]
        f_cost = self.fuzzy.at[generator, 'f_cost']._interp_membership(self.fuzzy.at[generator, 'f_cost'].fuzzy_set['cost'], cost)
        
        return f_cost
    
    def minimum_up(self, generator, hour):
        
        minup = self.cons_set.up_last(generator, hour)
        f_minup = self.fuzzy.at[generator, 'f_minup']._interp_membership(self.fuzzy.at[generator, 'f_minup'].fuzzy_set['minup'], minup)
            
        return f_minup
    
    def minimum_down(self, generator, hour):
        
        mindown = self.cons_set.down_last(generator, hour)
        f_mindown = self.fuzzy.at[generator, 'f_mindown']._interp_membership(self.fuzzy.at[generator, 'f_mindown'].fuzzy_set['mindown'], mindown)
        
        return f_mindown
    
    def startup(self, generator, hour):
        
        stup = self.net.net.res_startup.at[hour, generator]
        f_stup = self.fuzzy.at[generator, 'f_startup']._interp_membership(self.fuzzy.at[generator, 'f_startup'].fuzzy_set['startup'], stup)

        return f_stup
        
    def state_fc(self, generator, hour):
        
        m_fcost = self.net.net.res_priority.loc[hour, :].max()
        gen_fcost = self.net.net.res_priority.at[hour, generator]
        
        return m_fcost/ gen_fcost
        
    def create_reward(self, generator, hour, action):
        
        f_cost = self.cost_set.fixed_cost(generator, hour)
        stup_cost = self.cost_set.startup_cost(generator, hour)
        reward = (f_cost+stup_cost) * action + self.cons_set.penalty
        
        return reward
    
    def create_treward(self, generator, hour):
        
        _, hour_id = hour.split('-')
        hour_id = int(hour_id)
        
        f_cost = self.cost_set.cost(hour)
        stup_cost = self.cost_set.stup(hour)
        t_cost = f_cost + stup_cost
        total = sum(t_cost)
        penalty = self.cons_set.penalty
        tt = np.append(t_cost, total)
        ttp = np.append(tt, penalty)
        self.net.net.res_reward.loc[hour, :] = ttp
        
        return total+penalty
    