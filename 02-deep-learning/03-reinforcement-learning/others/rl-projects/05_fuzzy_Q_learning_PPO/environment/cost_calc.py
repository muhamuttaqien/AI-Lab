import numpy as np
import pandas as pd

class Cost_calc(object):
    
    def __init__(self, net):
        
        self.net = net.net
        self.lamda = 1000
        
    def f_cost(self, hour):
        
        u = self.net.res_generator.loc[:, 'u_max'].values
        ag = self.net.res_generator.loc[:, 'a_g'].values
        bg = self.net.res_generator.loc[:, 'b_g'].values
        cg = self.net.res_generator.loc[:, 'c_g'].values
        
        fixed_cost = ag*u**2 + bg*u + cg
        self.net.res_cost.loc[hour, :] = fixed_cost
        
        return fixed_cost
    
    def cost(self, hour):
        
        u = self.net.res_generation.loc[hour, :].values
        ag = self.net.res_generator.loc[:, 'a_g'].values
        bg = self.net.res_generator.loc[:, 'b_g'].values
        cg = self.net.res_generator.loc[:, 'c_g'].values
        
        schedule = self.net.res_on_off_schedule.loc[hour, :].values
        fixed_cost = ag*u**2 + bg*u + cg*schedule
        self.net.res_cost.loc[hour, :] = fixed_cost
        
        return fixed_cost
        
    def fixed_cost(self, generator, hour):
        
        _, gen_id = generator.split('-')
        gen_id = int(gen_id)
        
        u = self.net.res_generation.at[hour, generator]
        ag = self.net.res_generator.at[gen_id, 'a_g']
        bg = self.net.res_generator.at[gen_id, 'b_g']
        cg = self.net.res_generator.at[gen_id, 'c_g']
        
        fixed_cost = ag*u**2 + bg*u + cg
        
        return fixed_cost
    
    def stup_cost(self, hour):
        
        hoff = self.net.res_down.loc[hour, :].values
        hdown = self.net.res_generator.loc[:, 't_down'].values
        hcold = self.net.res_generator.loc[:, 't_cold'].values
        
        for i in range(self.net.gen_len):
            gen_id = i
            generator = 'gen-' + str(i)
            startup_cost = 0
            
            if hoff[i] <= hdown[i] + hcold[i]:
                need_stup = int(self.net.res_down.at[hour, generator] >= 1)
                startup_cost = self.net.res_generator.at[gen_id, 'hs_up'] * need_stup
            else:
                need_stup = int(self.net.res_down.at[hour, generator] >= 1)
                startup_cost = self.net.res_generator.at[gen_id, 'cs_up'] * need_stup
            
            self.net.res_startup.at[hour, generator] = startup_cost
    
        return self.net.res_startup.loc[hour, :]
    
    def stup(self, hour):
        
        hoff = self.net.res_down.loc[hour, :].values
        hdown = self.net.res_generator.loc[:, 't_down'].values
        hcold = self.net.res_generator.loc[:, 't_cold'].values
        
        schedule = self.net.res_on_off_schedule.loc[hour, :].values
        st_cost = []
        
        for i in range(self.net.gen_len):
            
            gen_id = i
            generator = 'gen-' + str(i)
            startup_cost = 0
            
            if hoff[i] <= hdown[i] + hcold[i]:
                need_stup = int(self.net.res_down.at[hour, generator] >= 1)
                startup_cost = self.net.res_generator.at[gen_id, 'hs_up'] * need_stup * schedule[gen_id]
            else:
                need_stup = int(self.net.res_down.at[hour, generator] >= 1)
                startup_cost = self.net.res_generator.at[gen_id, 'cs_up'] * need_stup * schedule[gen_id]
                
            st_cost.append(startup_cost)
            
        return st_cost
    
    def startup_cost(self, generator, hour):
        
        _, gen_id = generator.split('-')
        gen_id = int(gen_id)
        
        hoff = self.net.res_down.at[hour, generator]
        hdown = self.net.res_generator.at[gen_id, 't_down']
        hcold = self.net.res_generator.at[gen_id, 't_cold']
        startup_cost = 0
        
        if hoff <= hdown+hcold:
            need_stup = int(self.net.res_up.at[hour, generator] < 2)
            startup_cost = self.net.res_generator.at[gen_id, 'hs_up'] * need_stup
        else:
            need_stup = int(self.net.res_up.at[hour, generator] < 1)
            startup_cost = self.net.res_generator.at[gen_id, 'cs_up'] * need_stup

        return startup_cost
    
    def hour_cost(self, generator, hour):
        
        _, gen_id = generator.split('-')
        gen_id = int(gen_id)
        
        _, hour_id = hour.split('-')
        hour_id = int(hour_id)
        
        fule_cost = self.net.res_cost.loc[hour, :].sum()
        stup_cost = self.net.res_startup.loc[hour, :].sum()
        t_demand = self.net.res_load.at[hour_id, 'amount']
        u_generation = self.net.res_generation.loc[hour, :].sum()
        
        pen_generation = 0
        h_cost = fule_cost + stup_cost + self.lamda*(t_demand-u_generation) + pen_generation
        
        return h_cost
    
    def penalty_cal(self):
        pass
    