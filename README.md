# Privacy-Acquisition-Algorithm-in-Federated-Learning
This is my undergraduate graduation project about a method to extract private information from federated learning. It is based on previous FILM attack(https://github.com/Princeton-SysML/FILM).

# Introduction
The scenario of my experiment is that there is a curious eavesdropper who is a part of the clients in federated learning and has access to the word embedding gradients. Whenever the server is sending word embedding gradients to clients, the eavesdropper can have access to the gradients. Then the eavesdropper can use inversion attack to deduce the words that are highly likely to be in the training data and extract them.

![Introduction](images/intro.png)

