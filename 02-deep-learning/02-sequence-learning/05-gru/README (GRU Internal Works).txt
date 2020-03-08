# ---------------------------
# GRUs (Gated Recurrent Unit)
# ---------------------------
1. GRUs are quite faster and less extensive of computational resources yet still very new then LSTMs

2. GRUs got rid of the cell state and used only the one hidden state to transfer short-term and long-term information

3. GRUs also only have two gates, a reset gate and update gate

4. GRUs are not washing out the new input every single time but keeps the relevant information and passes it down to the next time steps of the network

5. GRUs' update gate acts similar to the forget and input gate of an LSTM while it decides what information to throw away and what new information to add

6. GRUs' reset gate is another gate to decide how much past information to forget

7. RESET GATE is achieved by multiplying the previous hidden state (ht-1) and current input (xt) with their respective weights (Whidden, Winput) and summing them before passing the sum through a sigmoid function | Greset = sigmoid(  Whidden-reset . ht-1 + Winput-reset . xt ) | r = tanH(Greset x ( Whidden . ht-1 ) + ( winput . xt ) ) | x = Hadamard product

8. UPDATE GATE is to help the model determine how much of the past information stored in the previous hidden state needs to be retrained for the future use | Gupdate = sigmoid( Whidden-update . ht-1 + Winput-update . xt ) | u = Gupdate x ht-1 | x = Hadamard product

9. GRUs consits combining process for RESET GATE and UPDATE GATE and this process will produce new hidden state transfering short-term and long-term information | ht = r x (1 - Gupdate) + u | x = Hadamard product

10. To review, RESET GATE is responsible for deciding which portions of the previous hidden state are to be combined with the current input to propose a new hidden state while UPDATE GATE is responsible for determining how much of the previous hidden state is to be retained and what portion of the new proposed hidden state (derived from the RESET GATE) is to be added to the final hidden state

11. GRUs, similar with its older sibling LSTMs, keep most of the existing hidden state while adding new content on top of it (not replacing the entire content of the hidden state at each time step like RNNs do)

12. The role of the UPDATE GATE in GRUs is very similar to INPUT and FORGET GATES in LSTMs however, the control of new memory content added to the network differs between these two

13. In LSTMs, while FORGET GATE determines which part of the previous cell state to retain, INPUT GATE determines the amount of the new memory to be added and these two gates are independent of each other meaning that the amount of new information added through INPUT GATE is completely independent of the information retained through FORGET GATE

14. As for GRUs, UPDATE GATE is responsible for determining which information from the previous memory to retain and is also responsible for controlling the new memory to be added meaning that the retention of previous memory and addition of new information to the memory in GRUs is NOT independent