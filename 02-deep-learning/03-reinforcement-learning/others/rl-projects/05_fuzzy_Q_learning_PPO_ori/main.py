import numpy as np 
import pandas as pd 
from collections import deque
from environment.environment import Environment
from ppo.main import Policy
from ppo.memory import Memory
import matplotlib.pyplot as plt

memory=Memory(240)
env=Environment()
agents=[]

agent=Policy(env.state_len,env.action_len,"agent")


# reset environment
rewards=[]
for _ in range(5000):
    env.reset()
    state=np.zeros(env.state_len)
    for j in range(24):
        hour="Hour-"+str(j)
        actions=[]
        for i in range(10):
            genn="gen-"+str(i)
            act=agent.get_action(state)
            action=np.argmax(act)
            actions.append(action)
            env.net.net.res_on_off_schedule.loc[hour,genn]=action
            n_state,reward=env.step(genn,hour,action)
            if i==9:
                gen_end=True
            else:
                gen_end=False
            if j==23:
                done=True
            else:
                done=False
            memory.pre_store(state, act, reward, n_state, done)
            if gen_end:
                schedule=env.net.net.res_use_schedule.loc[hour,:].values
                on_off=env.net.net.res_on_off_schedule.loc[hour,:].values
                for id,(sstate, saction, re, sstate_next, sdone) in enumerate(memory.memo):
                    if schedule[id]==1 and on_off[id]==1:
                        agent.memorize(sstate, saction, reward, sstate_next, sdone)
                    elif schedule[id]==0 and on_off[id]==1:
                        agent.memorize(sstate, saction, -reward*0.8, sstate_next, sdone)
                    else:
                        agent.memorize(sstate, saction,  reward*0.01, sstate_next, sdone)
                memory.memo=deque(maxlen=96)
            state=n_state
    print("episode action",_,reward)
    rewards.append(reward)
        
print("schedule after",env.net.net.res_on_off_schedule)
print("generation after",env.net.net.res_generation)
print("generation after",env.net.net.res_generation.sum(axis=1))
print("cost after",env.net.net.res_cost)
print("penalty after",env.net.net.res_reward)
plt.plot(rewards)
plt.show()