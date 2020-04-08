import math

def log_prob_density(x, mu, std):
    
    log_prob_density = -(x - mu).pow(2) / (2 * std.pow(2)) - 0.5 * math.log(2 * math.pi)
    return log_prob_density.sum(1, keepdim=True)

def surrogate_loss(actor, advantages, states, old_policy, actions, batch_index):
    
    mu, std = actor(states)
    new_policy = log_prob_density(actions, mu, std)
    old_policy = old_policy[batch_index]
    
    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advantages
    
    return surrogate_loss, ratio
