# ---------------------------------
# RNNs (Recurrent Neural Networks)
# ---------------------------------
1. In RNNs, The prior ht is initialized by 0 values

2. There is Whh with weights randomly initialized

3. To implement RNNs, the technique can be one-to-one, one-to-many, many-to-one, many-to-many and many-to-many

4. In RNNs, we may or may not have outputs at each time step

5. RNNs output produced can also be fed back into the model at the next time step if necessary called looping mechanisms

6. RNNs, theoretically, hidden state is init for each time step but in practice it can be done for every batch or epoch and it worked

7. RNNs typically treat the full sequence (word) as one training example so the total error is just the sum of the errors at each time step (chars)

8. All of the weights are actually the same as that RNNs cell is essentially being reused throughout the process and only the input data and hidden state carried forward are unique at each time step

9. RNNs gradient is calculated for each time step with respect to the weight parameter and then combine the gradients of the error for all time steps (If there are 100s of time steps, this would basically take really long for the network to converge suffering vanishing gradients and the values are almost zero)

10. RNNs are forgetful so they cannot handle long dependency like ‘The man who ate my pizza has purple hair’ sentence

11. RNNs formula for generating hidden state is Ht = tanH( Whh*Ht-1 + Wxh*X )

12. RNNs hidden state init in batch shape respects to input’s batch shape

13. RNNs input size and output size is as same as dictionary/ vocabulary size of the tasks

14. In practice we can use until 300 memory cells/ units and more than 1 RNNs layer