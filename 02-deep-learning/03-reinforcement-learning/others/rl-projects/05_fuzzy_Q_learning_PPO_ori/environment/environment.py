import numpy as np 
import pandas as pd 
from simulator.network import *
from simulator.excel_read_write import Net_extend
from .cost_calc import Cost_calc
from .cons_imp import Cons_imp
from .fuzzy import Fuzzy

class Environment():

    def __init__(self):
        self.net=Network(new=True)
        Net_extend(self.net)
        self.next_state=[]
        self.action=["on","off"]
        self.state=[0,0,0,0,0,0]
        self.state_len=len(self.state)
        self.action_len=len(self.action)
        self.fuzz=self.net.net.res_fuzzy
        self.cost_set=Cost_calc(self.net)
        self.cons_set=Cons_imp(self.net)
        self.reset() #for fuzzy cost setup
        self.create_fuzzy()

    def reset(self):
        self.net.reset_results()
        gen="gen-0"
        _,gen_id=gen.split("-")
        gen_id=int(gen_id)
        self.hourly_setup(gen,"Hour-0")
        val=self.net.net.res_generator.loc[:,"t_initial"].values>0
        vals=val.astype(int)
        self.net.net.res_up.loc["Hour-0",:]=self.net.net.res_generator.loc[gen_id,"t_initial"].T *vals

    def hourly_setup(self,gene,hour): 
        if gene=="gen-0":
            self.cost_set.f_cost(hour)
            self.cons_set.priority_list(hour)
            self.cons_set.sefty_spining(hour)
        if gene=="gen-9":
            self.cons_set.set_generatation(hour)
            self.cons_set.minimum_up_down(hour)
            self.cost_set.stup_cost(hour)

    def step(self,generator,hour,action):
        self.hourly_setup(generator,hour)
        state=self.create_state(generator,hour,action)
        reward=self.create_reward(generator,hour,action)
        if generator=="gen-9":
            reward=self.create_treward(generator,hour)
        return state,reward

    def create_state(self,generator,hour,action):
        state=[]
        state.append(self.capacity(generator,hour))
        state.append(self.cost(generator,hour))
        state.append(self.minimum_up(generator,hour))
        state.append(self.minimum_down(generator,hour))
        state.append(self.startup(generator,hour))
        state.append(self.state_fc(generator,hour))
        return state

    def create_fuzzy(self):
        fuss=self.net.net.fuzzy
        cost=self.net.net.res_cost
        mindown=self.net.net.res_generator
        h="Hour-0"
        for i in range(self.net.net.gen_len):
            generator="gen-"+str(i)
            self.fuzz.at[generator,'f_gen']=Fuzzy([0,fuss.at[i,"left0"],fuss.at[i,"span"]])
            self.fuzz.at[generator,'f_gen'].trapmf([fuss.at[i,"right0"],fuss.at[i,"right0"],fuss.at[i,"right"],fuss.at[i,"right"]],"g_low")
            self.fuzz.at[generator,'f_gen'].trapmf([fuss.at[i,"right"],fuss.at[i,"right"],fuss.at[i,"left0"],fuss.at[i,"left0"]],"g_high")
            ##for generation plot
            #self.fuzz.at[generator,'f_gen']._plot(self.fuzz.at[generator,'f_gen'].fuzzy_set["g_low"],self.fuzz.at[generator,'f_gen'].fuzzy_set["g_high"])
            self.fuzz.at[generator,'f_cost']=Fuzzy([cost.loc[h,:].min()-50,cost.loc[h,:].max()+50,50])
            self.fuzz.at[generator,'f_cost'].trinf([cost.loc[h,:].min(),cost.loc[h,:].max(),cost.loc[h,:].max()],"cost")
            ##cost plot
            #self.fuzz.at[generator,'f_cost']._plot(self.fuzz.at[generator,'f_cost'].fuzzy_set["cost"])
            self.fuzz.at[generator,'f_minup']=Fuzzy([0,fuss.at[i,"minup"]+10,1])
            self.fuzz.at[generator,'f_minup'].trinf([0,fuss.at[i,"minup"],fuss.at[i,"minup"]],"minup")
            ##plot minimum up 
            #self.fuzz.at[generator,'f_minup']._plot(self.fuzz.at[generator,'f_minup'].fuzzy_set["minup"])
            self.fuzz.at[generator,'f_mindown']=Fuzzy([0,mindown.at[i,"t_down"]+10,1])
            self.fuzz.at[generator,'f_mindown'].trinf([0,mindown.at[i,"t_down"],mindown.at[i,"t_down"]+10],"mindown")
            ##plot minimum down
            #self.fuzz.at[generator,'f_mindown']._plot(self.fuzz.at[generator,'f_mindown'].fuzzy_set["mindown"])
            self.fuzz.at[generator,'f_startup']=Fuzzy([0,mindown.at[i,"cs_up"]+10,mindown.at[i,"hs_up"]])
            self.fuzz.at[generator,'f_startup'].trinf([0,mindown.at[i,"cs_up"],mindown.at[i,"cs_up"]+10],"startup")
            ##plot minimum down
            #self.fuzz.at[generator,'f_startup']._plot(self.fuzz.at[generator,'f_startup'].fuzzy_set["startup"])

    def capacity(self,generator,hour):
        # generation=self.net.net.res_generation.at[hour,generator]
        # low_generation=self.fuzz.at[generator,'f_gen']._interp_membership(self.fuzz.at[generator,'f_gen'].fuzzy_set["g_low"],generation)
        # high_generation=self.fuzz.at[generator,'f_gen']._interp_membership(self.fuzz.at[generator,'f_gen'].fuzzy_set["g_high"],generation)
        _,hour_id=hour.split("-")
        hour_id=int(hour_id)
        _,gen_id=generator.split("-")
        gen_id=int(gen_id)
        t_demand=self.net.net.res_load.at[hour_id,"amount"]
        gen_umax =self.net.net.res_generator.loc[gen_id,"u_max"]
        gen=gen_umax/t_demand
        return gen

    def cost(self,generator,hour):
        cost=self.net.net.res_cost.at[hour,generator]
        f_cost=self.fuzz.at[generator,'f_cost']._interp_membership(self.fuzz.at[generator,'f_cost'].fuzzy_set["cost"],cost)
        return f_cost

    def minimum_up(self,generator,hour):
        minup=self.cons_set.up_last(generator,hour)
        f_minup=self.fuzz.at[generator,'f_minup']._interp_membership(self.fuzz.at[generator,'f_minup'].fuzzy_set["minup"],minup)
        return f_minup

    def minimum_down(self,generator,hour):
        mindown=self.cons_set.down_last(generator,hour)
        f_mindown=self.fuzz.at[generator,'f_mindown']._interp_membership(self.fuzz.at[generator,'f_mindown'].fuzzy_set["mindown"],mindown)
        return f_mindown

    def startup(self,generator,hour):
        stup=self.net.net.res_startup.at[hour,generator]
        f_stup=self.fuzz.at[generator,'f_startup']._interp_membership(self.fuzz.at[generator,'f_startup'].fuzzy_set["startup"],stup)
        return f_stup

    def state_fc(self,generator,hour):
        m_fcost=self.net.net.res_priority.loc[hour,:].max()
        gen_fcost=self.net.net.res_priority.at[hour,generator]
        return m_fcost/gen_fcost

    def create_reward(self,generator,hour,action):
        f_cost=self.cost_set.fixed_cost(generator,hour)
        stup_cost=self.cost_set.startup_cost(generator,hour)
        reward=(f_cost+stup_cost)*action+self.cons_set.penalty
        return reward

    def create_treward(self,generator,hour):
        _,hour_id=hour.split("-")
        hour_id=int(hour_id)
        f_cost=self.cost_set.cost(hour)
        stup_cost=self.cost_set.stup(hour)
        t_cost=f_cost+stup_cost
        total=sum(t_cost)
        penlaty=self.cons_set.penalty
        tt=np.append(t_cost,total)
        ttp=np.append(tt,penlaty)
        self.net.net.res_reward.loc[hour,:]=ttp
        return total+penlaty