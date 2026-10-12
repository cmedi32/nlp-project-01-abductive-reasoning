# Expectations

## Initial Expectations (as of 6.10.2026 / official project start, no more experimentation) PROJECT 1

## Using embeddings with and without RNN - static or non recurrent

Models are simple and have no abstract idea of the context.
Models simply rely on the most probable next word trained on data representing a very small world

Accuracy Scores - i would expect it go up until around 60 - 70% given that at this time and day i have not yet investigated the dataset in detail, just based on what i would expect from such "simple" neural networks.

A modern small transformer with 2-3B weights i would expect to have a near 99.9% accuracy.

I have never trained a ML model from scratch before, so the first one here is actualy "the first one". maybe a language based one isn't the simplest to start with but i will need to figure it out.

TODOS:

- i will learn about the split of the set and how to apply it for training, forward pass, and evaluation.. 
- i will take the data set, run it through pre processing.
- then i will figure out how to embed it into vector space.
- then i will figure out the hyperparameters of word2vec and see what i have to take right over to my neural network, for me these are still 2 different things (maybe i didn't get it.. will see in the implementation)
- then i will try to orchestrate a forward pass, investigate loss and backpropagated my changes to the weights (like i said, i have never done this before)
- i assume all of the stuff should be doable with the libraries at hand - i don't think i have to juggle around tensor math.. but will see...
- once i figured out how to backpropagate and actualy "train" a model, i will try saving the weights and hyperparemeters as well as the results of inference using the test dataset on weights and biases.. i have never done that too before.

I will try to do it all on my workstation - running an older nvidia architecture, rtx 2000 series. I would expect that this is more than enough for the non transformer based solutions, but will see.

### Variations of the solutions

whatever comes next, i will experiment around with whats documented and see how far i get with the results

1. simple preprocessing, simple w2v with LogReg and meaning the result vectors - just to build the simplest model possible end to end i expected nothing eg. worst than just taking the mean or using a random classification.

2. same preprocessing, simple w2v, same lookup same logic but now with an actual multi layer model (MLP), i want to try appying and experimenting with the learning rate and log it all to weights and biases i expect to get up to maybe 70% compared to the fifty fifty before.

3. i wan't to try glove word embedding, compare the OOV tokens to w2v and see what works better also weighting better using tfidf -  i expect a small upstep.. maybe 75%. From variant 2 i learned that simply averaging the vectors and mean them into one eg h1, h2, r1, r2 losses to much information. the loss function stops after very few iterrations already. I also found out in ai research that the ceiling for non recurrent models is much lower.

4. Try using tfidf instead of simple concatenation - this is what i think will get the maximum of maybe 52-53%, i don't think there is any more to do without recurrent networks

## Using embeddings with RNNs - recurrent implementations

ROADMAP (first draft):

1	LSTM, unidirectional, 1 layer, last hidden state	RNN 50–58, Adv 16
2	LSTM → GRU	RNN 62–66
3	Unidirectional → Bidirectional	Adv 8–14
4	1 layer → 2 layers	RNN 40
5	Last hidden state → Attention pooling	Adv 29–37