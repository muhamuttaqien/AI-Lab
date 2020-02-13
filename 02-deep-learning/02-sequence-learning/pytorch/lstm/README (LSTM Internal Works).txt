# -------------------------------
# LSTMs (Long Short-Term Memory)
# -------------------------------
1. For the simple short-term-memory sentence like ‘The color of the sky is ___‘, RNNs turn out to be quite effective

2. The limitation of RNNs is they transforms the existing information completely and there is no consideration for ‘important’ and ‘not so important’ information

3. LSTMs network is comprised of different memory blocks called cells 

4. LSTMs’ cell state contains long-term memory computed by forget & input gates and hidden state contains short-term memory computed by output memory and these gates called 3 major gating mechanisms (forget, input, output)

5. FORGET GATE is responsible for forget long-term memory/not, long-term memory carried by cell state will be considered whether to remember/ keep or forget/ remove the information from the cell sate | Gforget = sigmoid( Wforget . ( ht-1, xt ) + Bforget )

6. In FORGET GATE, sentence example is ‘Bob is a nice person. Dan on the other hand is evil’ while the gate forgets the subject ‘Bob’ and replaces it with the new subject ‘Dan’

7. INPUT GATE is responsible for adjust long-term memory/not long-term memory carried by cell state will be considered whether to adjust or not based on vector created from the tanH | Ginput1 = sigmoid( Winput1 . (ht-1, xt) + Binput1 ) | Ginput2 = tanH( Winput2 . (ht-1, xt) + Binput2 ) | Ginput1 * Ginput2

8. In INPUT GATE, sentence example is ‘Bob knows swimming. He told me over the phone that he had served the navy for 4 long years’ while the gate set ‘he told over the phone’ as less important information or might be redundant and can be ignored

9. OUTPUT GATE is responsible for select useful information from newly-computed long-term memory/not and short-term memory since not all information that runs along the cell state is fit for being output/ prediction and new short-term memory | Goutput1 = sigmoid( Woutput1 . (ht-1, xt) + Boutput1 ) | Goutput2 = tanH( Woutput2 . Cellt + Boutput2 ) | Goutput1 * Goutput2 

10. In OUTPUT GATE, sentence example is ‘Bob fought single handedly with the enemy and died for his country. For his contributions brave’ while the gate has a strong tendency of the answer being a noun and thus ‘Bob’ could be an output

11. The one of disadvantage of LSTMs is the difficulty of training them meaning it takes a lot of time and system resources (hardware constraint)