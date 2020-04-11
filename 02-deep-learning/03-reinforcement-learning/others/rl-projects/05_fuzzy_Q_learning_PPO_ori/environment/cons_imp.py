import numpy as np 
import pandas as pd 

class Cons_imp:
    def __init__(self,net):
        self.net=net.net
        self.penalty=0

    def priority_list(self,hour):
        """
        to calculate priority list
        hour fc--> hourly fule cost of generation
        gen umax--> generators maximum production capacity
        vals-->  fule cost / generator max capacity
        temp--> sorted to vals
        res--> index of sorted list
        """
        hourly_fc=self.net.res_cost.loc[hour,:].values
        gen_umax =self.net.res_generator.loc[:,"u_max"].values 
        vals=hourly_fc/gen_umax
        self.net.res_priority.loc[hour,:]=vals

    def set_generatation(self,hour):
        """
        to calculate priority list
        hour fc--> hourly fule cost of generation
        gen umax--> generators maximum production capacity
        vals-->  fule cost / generator max capacity
        temp--> sorted to vals
        res--> index of sorted list
        """
        self.penalty=0
        t_demand=0
        _,hour_id=hour.split("-")
        hour_id=int(hour_id)
        hourly_fc=self.net.res_priority.loc[hour,:].values
        schedule=self.net.res_on_off_schedule.loc[hour,:].astype(int)
        temp = sorted(hourly_fc)    
        res = [temp.index(i) for i in hourly_fc]
        t_demand=self.net.res_load.at[hour_id,"amount"]
        demand_res=t_demand+t_demand*self.net.spining_reserve
        gen_umax=0
        balance=0
        for id in res:
            generator="gen-"+str(id)
            if demand_res>gen_umax and schedule[id]==1:
                balance=demand_res-gen_umax
                if balance>self.net.res_generator.at[id,"u_max"]:
                    self.net.res_use_schedule.at[hour,generator]=1
                    self.net.res_generation.at[hour,generator]=self.net.res_generator.at[id,"u_max"]
                else:
                    if self.net.res_generator.at[id,"u_min"]>(gen_umax-demand_res):
                        self.net.res_use_schedule.at[hour,generator]=1
                        self.penalty=(self.net.res_generator.at[id,"u_min"]-(gen_umax-demand_res))**2
                        self.net.res_generation.at[hour,generator]=self.net.res_generator.at[id,"u_min"]
                        break
                    else:
                        self.net.res_use_schedule.at[hour,generator]=1
                        self.net.res_generation.at[hour,generator]=gen_umax-demand_res
                        break
                gen_umax +=self.net.res_generator.at[id,"u_max"]
        if self.net.res_generation.loc[hour,:].sum()<demand_res:
            self.penalty+=(demand_res-self.net.res_generation.loc[hour,:].sum())*1000
    def sefty_spining(self,hour):
        """
        to calculate sefty spining
        gen umax--> generators maximum production capacity
        t_demand-->  demand at that hour
        spining reserve--> percentage of spinin reserve in model
        schedule--> the generator on off schedule pre defined 
        ESR --> Expeced Spining Reserve 
        ESR--> expected generation - demand - spining reserve
        """
        _,hour_id=hour.split("-")
        hour_id=int(hour_id)
        gen_umax =self.net.res_generator.loc[:,"u_max"].values
        t_demand=self.net.res_load.at[hour_id,"amount"]
        spining_reserve=self.net.spining_reserve
        schedule=self.net.res_on_off_schedule.loc[hour,:].values
        ESR=sum(gen_umax*schedule)-t_demand-t_demand*spining_reserve
        return ESR  

    def power_bal(self,hour):
        """
        to calculate minimum up down hour
        hour_id--> int of hour
        t_demand-->  demand at that hour
        u_generation--> total generation at the hour
        schedule--> the generator on off schedule pre defined
        output--> boolean value of demand is less or equal to production
        """
        _,hour_id=hour.split("-")
        hour_id=int(hour_id)
        schedule=self.net.res_on_off_schedule.loc[hour,:].values
        u_generation=self.net.res_generation.loc[hour,:].values
        t_demand=self.net.res_load.at[hour_id,"amount"]
        return sum(schedule*u_generation)>=t_demand

    def gen_limit(self,hour,generator):
        """
        to calculate sefty spining
        gen umax--> generators maximum production capacity
        gen umin--> generators minimum production capacity
        schedule--> the generator on off schedule pre defined 
        u_generation-->generation at the hour
        """
        _,hour_id=hour.split("-")
        hour_id=int(hour_id)
        _,gen_id=generator.split("-")
        gen_id=int(gen_id)
        gen_umax =self.net.res_generator.at[gen_id,"u_max"]
        gen_umin =self.net.res_generator.at[gen_id,"u_min"]
        schedule=self.net.res_on_off_schedule.at[hour,generator]
        u_generation=self.net.res_generation.loc[hour,generator]
        return ((gen_umin*schedule)<=u_generation<=(gen_umax*schedule))

    def minimum_up_down(self,hour):
        """
        to calculate minimum up down hour
        lhour--> last hour 
        u_lhour--> str of last hour
        lhour on--> hourly generator on data 
        lhour off--> hourly generator off data 
        hour_on--> generator on time hour
        hour_off --> generator off time hour
        """
        _,hour_id=hour.split("-")
        hour_id=int(hour_id)
        lhour=self.last_hour(hour)
        u_lhour="Hour-"+str(lhour)
        lhour_on=self.net.res_up.loc[u_lhour,:].values
        if hour_id==0:
            val=self.net.res_generator.loc[:,"t_initial"].values>0
            vals=val.astype(int)
            lhour_on=self.net.res_generator.loc[hour_id,"t_initial"].T *vals
        lhour_off=self.net.res_down.loc[u_lhour,:].values
        schedule=self.net.res_on_off_schedule.loc[hour,:].values
        hour_on=(1+lhour_on)*schedule
        hour_off=(1+lhour_off)*(1-schedule)
        self.net.res_up.loc[hour,:]=hour_on
        self.net.res_down.loc[hour,:]=hour_off

    def up_last(self,generator,hour):
        """
        to calculate minimum up down hour
        lhour--> last hour 
        u_lhour--> str of last hour
        lhour on--> hourly generator on data 
        lhour off--> hourly generator off data 
        hour_on--> generator on time hour
        hour_off --> generator off time hour
        """
        _,hour_id=hour.split("-")
        hour_id=int(hour_id)
        _,gen_id=generator.split("-")
        gen_id=int(gen_id)
        lhour=self.last_hour(hour)
        u_lhour="Hour-"+str(lhour)
        if hour_id==0:
            if (self.net.res_generator.at[gen_id,"t_initial"]>0):
                lhour_on= self.net.res_generator.at[gen_id,"t_initial"]
            else:
                lhour_on=0
        else:
            lhour_on=self.net.res_up.at[u_lhour,generator]
        return lhour_on

    def down_last(self,generator,hour):
        """
        to calculate minimum up down hour
        lhour--> last hour 
        u_lhour--> str of last hour
        lhour on--> hourly generator on data 
        lhour off--> hourly generator off data 
        hour_on--> generator on time hour
        hour_off --> generator off time hour
        """
        _,hour_id=hour.split("-")
        hour_id=int(hour_id)
        _,gen_id=generator.split("-")
        gen_id=int(gen_id)
        lhour=self.last_hour(hour)
        u_lhour="Hour-"+str(lhour)
        if hour_id==0:
            lhour_off=0
        else:
            lhour_off=self.net.res_down.at[u_lhour,generator]
        return lhour_off
        

    def last_hour(self,hour):
        """
        to calculate last hour
        net.gen_len --> number of generator in the model
        """
        lhour=None
        _,hour_id=hour.split("-")
        hour_id=int(hour_id)
        if hour_id==0:
            lhour=self.net.gen_len-1
        else:
            lhour=hour_id-1
        return lhour

