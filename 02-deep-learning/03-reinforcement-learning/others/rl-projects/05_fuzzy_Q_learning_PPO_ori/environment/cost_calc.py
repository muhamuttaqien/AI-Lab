import numpy as np 
import pandas as pd 

class Cost_calc:
    def __init__(self,net):
        """
            lamda for penalty factor for unit MW
        """
        self.net=net.net
        self.lamda=1000

    def f_cost(self,hour):
        """
        bsed on 1 unit production
        hourly group calculation
        to calculate the fixed cost
        FC(g)=agU**2 + bgU + cg
        """
        u        =self.net.res_generator.loc[:,"u_max"].values
        ag       =self.net.res_generator.loc[:,"a_g"].values
        bg       =self.net.res_generator.loc[:,"b_g"].values
        cg       =self.net.res_generator.loc[:,"c_g"].values
        fixed_cost=ag*u**2+bg*u+cg
        self.net.res_cost.loc[hour,:]=fixed_cost
        return fixed_cost

    def cost(self,hour):
        """
        bsed on 1 unit production
        hourly group calculation
        to calculate the fixed cost
        FC(g)=agU**2 + bgU + cg
        """
        u        =self.net.res_generation.loc[hour,:].values
        ag       =self.net.res_generator.loc[:,"a_g"].values
        bg       =self.net.res_generator.loc[:,"b_g"].values
        cg       =self.net.res_generator.loc[:,"c_g"].values
        schedule=self.net.res_on_off_schedule.loc[hour,:].values
        fixed_cost=ag*u**2+bg*u+cg*schedule
        self.net.res_cost.loc[hour,:]=fixed_cost
        return fixed_cost

    def fixed_cost(self,generator,hour):
        """
        based on total procduction
        hourly generator wise calculation
        to calculate the fixed cost
        FC(g)=agU**2 + bgU + cg
        """
        _,gen_id=generator.split("-")
        gen_id=int(gen_id)
        u        =self.net.res_generation.at[hour,generator]
        ag       =self.net.res_generator.at[gen_id,"a_g"]
        bg       =self.net.res_generator.at[gen_id,"b_g"]
        cg       =self.net.res_generator.at[gen_id,"c_g"]
        fixed_cost=ag*u**2+bg*u+cg
        #self.net.res_cost.at[hour,generator]=fixed_cost
        return fixed_cost

    def stup_cost(self,hour):
        """
        hourly group cost
        to calculate the start up cost
        hoff--> generator off 
        hon --> generator on 
        hdown--> generator off duration
        hup--> generator on duration
        hs--> if hoff<= hdown+hcold
        cs--> if hoff>= hdown+hcold
        """
        hoff        =self.net.res_down.loc[hour,:].values
        hdown       =self.net.res_generator.loc[:,"t_down"].values
        hcold       =self.net.res_generator.loc[:,"t_cold"].values
        for i in range(self.net.gen_len):
            gen_id=i
            generator="gen-"+str(i)
            startup_cost=0
            if hoff[i]<=hdown[i]+hcold[i]:
                need_stup=int(self.net.res_down.at[hour,generator]>=1)
                startup_cost=self.net.res_generator.at[gen_id,"hs_up"]*need_stup
            else:
                need_stup=int(self.net.res_down.at[hour,generator]>=1)
                startup_cost=self.net.res_generator.at[gen_id,"cs_up"]*need_stup
            self.net.res_startup.at[hour,generator]=startup_cost
        return self.net.res_startup.loc[hour,:]

    def stup(self,hour):
        """
        hourly group cost
        to calculate the start up cost
        hoff--> generator off 
        hon --> generator on 
        hdown--> generator off duration
        hup--> generator on duration
        hs--> if hoff<= hdown+hcold
        cs--> if hoff>= hdown+hcold
        """
        hoff        =self.net.res_down.loc[hour,:].values
        hdown       =self.net.res_generator.loc[:,"t_down"].values
        hcold       =self.net.res_generator.loc[:,"t_cold"].values
        schedule=self.net.res_on_off_schedule.loc[hour,:].values
        st_cost=[]
        for i in range(self.net.gen_len):
            gen_id=i
            generator="gen-"+str(i)
            startup_cost=0
            if hoff[i]<=hdown[i]+hcold[i]:
                need_stup=int(self.net.res_down.at[hour,generator]>=1)
                startup_cost=self.net.res_generator.at[gen_id,"hs_up"]*need_stup*schedule[gen_id]
            else:
                need_stup=int(self.net.res_down.at[hour,generator]>=1)
                startup_cost=self.net.res_generator.at[gen_id,"cs_up"]*need_stup*schedule[gen_id]
            st_cost.append(startup_cost)
        return st_cost

    def startup_cost(self,generator,hour):
        """
        to calculate the start up cost per hour per generator
        hoff--> generator off 
        hon --> generator on 
        hdown--> generator off duration
        hup--> generator on duration
        hs--> if hoff<= hdown+hcold
        cs--> if hoff>= hdown+hcold
        """
        _,gen_id=generator.split("-")
        gen_id=int(gen_id)
        hoff        =self.net.res_down.at[hour,generator]
        hdown       =self.net.res_generator.at[gen_id,"t_down"]
        hcold       =self.net.res_generator.at[gen_id,"t_cold"]
        startup_cost=0
        if hoff<=hdown+hcold:
            need_stup=int(self.net.res_up.at[hour,generator]<2)
            startup_cost=self.net.res_generator.at[gen_id,"hs_up"]*need_stup
        else:
            need_stup=int(self.net.res_up.at[hour,generator]<1)
            startup_cost=self.net.res_generator.at[gen_id,"cs_up"]*need_stup
        #self.net.res_startup.at[hour,:]=startup_cost
        return startup_cost

    def hour_cost(self,generator,hour):
        """
        to calculate the hourly total cost
        fule_cose--> hourly total agent cost
        stup cost--> hourly total agent start up cost
        t_demand-->  hourly demand
        u_generation--> total generation at the hour
        pen_generation--> total penalty by agent at this hour
        """
        _,gen_id=generator.split("-")
        gen_id=int(gen_id)
        _,hour_id=hour.split("-")
        hour_id=int(hour_id)
        fule_cost= self.net.res_cost.loc[hour,:].sum()
        stup_cost= self.net.res_startup.loc[hour,:].sum()
        t_demand=self.net.res_load.at[hour_id,"amount"]
        u_generation=self.net.res_generation.loc[hour,:].sum()
        pen_generation=0
        h_cost=fule_cost+stup_cost+self.lamda*(t_demand-u_generation)+pen_generation
        return h_cost

    def penalty_cal(self,generator):
        """
        to calculate the hourly total cost
        fule_cose--> hourly total agent cost
        stup cost--> hourly total agent start up cost
        t_demand-->  hourly demand
        u_generation--> total generation at the hour
        pen_generation--> total penalty by agent at this hour
        """
        pass

