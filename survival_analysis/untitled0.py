# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:32:27 2020

@author: gliu3
"""

# so these are your original hidden states for each direction
# in this case hidden size is 5, but this works for any size
direction_one_out = torch.tensor(range(5))
direction_two_out = torch.tensor(list(reversed(range(5))))
print('Direction one:')
print(direction_one_out)
print('Direction two:')
print(direction_two_out)

# before outputting they will be concatinated 
# I'm adding here batch dimension and sequence length, in this case seq length is 1
hidden = torch.cat((direction_one_out, direction_two_out), dim=0).view(1, 1, -1)
print('\nYour hidden output:')
print(hidden, hidden.shape)

# trivial case, reshaping for one hidden state
hidden_reshaped = hidden.view(1, 1, 2, -1)
print('\nReshaped:')
print(hidden_reshaped, hidden_reshaped.shape)

# This works as well for abitrary sequence lengths as you can see here
# I've set sequence length here to 5, but this will work for any other value as well
print('\nThis also works for more multiple hidden states in a tensor:')
multi_hidden = hidden.expand(5, 1, 10)
print(multi_hidden, multi_hidden.shape)
print('Directions can be split up just like this:')
multi_hidden = multi_hidden.view(5, 1, 2, 5)
print(multi_hidden, multi_hidden.shape)