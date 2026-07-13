---
title: "Can AI Design a Tactical Card Game from scratch? Part 1"
date: 2026-07-13
mathjax: true
tags: ["TCG", "Machine Learning", "ML", "AI", "Graph Neural Networks", "GNNs"]
draft: true
thumbnail: ""
---

## Intro

Several years back when I was working at Mediatonic, we were working on a Tactical Card Game based on Hearthstone 
which had recently launched to huge success. The designers were having a hard time creating a set of cards that 
worked well. As you can imagine, this is quite a hard task. The aim is to create a set of cards that have interesting 
synergies but don't have any cards that are too overpowered (OP). Meaning that these cards would be included in every 
deck and any players without these cards would be excessively disadvantaged. This is, of course, known as balancing 
the game, and for a game as complex as Hearthstone this is a very challenging task.

I thought that instead of trying to explore this huge space of possible card designs, it would be desirable to
create an agent that could play the game to a reasonable level, decide what cards to form into a deck and show that
the current set of cards obey the desired properties as detailed above. Even better, we could start with a completely 
unbalanced set of cards and get the agent to adjust the cards alongside learning to play such that the following 
criteria are met:

* no two cards should be too similar to one another
* synergies should exist between the cards to facilitate interesting deck building
* any subset of cards should form a deck that can play reasonably against a deck made from all cards

At the time, the state of the art in ML wasn't at a point where this was possible. But today thanks to some 
interesting advances, there's an approach that I think would work incredibly well and could pave the way for a new 
paradigm of making games.

## Current limitations of Reinforcement Learning

The naive approach here would be to train an agent to play the game based on the current set of cards, update the cards 
based on the metrics above, and repeat until all criteria have been optimised. The major issue here is that training an 
agent to play even a simple version is computationally expensive. On top of this, the agent will 
need to start from scratch after each update of the card deck we see this approach is totally unfeasible.

Instead, we need to train an agent that can play a generalised version of the game, so that when the agent sees a 
card it has never seen before, it can still play strategically.

## Graph Neural Networks

Enter Graph Neural Networks (GNNs). These are a type of neural network that are applied to data that is represented 
as a graph. For example, in a social network, where the nodes are people and the edges are friendships, what's the 
chance a new edge will be formed between two people? GNNs work by applying a transformation to each 
node and to the edges that allow information to be propagated throughout the graph. This then can be used to make 
prediction on a node, edge or graph level.

This approach has been used very successfully to train agents in Go and Chess, both allowing agents to train far 
more rapidly but starting on smaller board sizes and building up, and also to train to a higher level than previous 
more rigid agents such as AlphaGo.

The setup for these games roughly follows this approach is described here for Chess: each node is a square, which 
contains which piece type it contains. Edges represent the legal moves from a given square. A global node can 
therefore be used to calculate the value state of a given position, and a policy network can come up with a softmax 
distribution to decide which move to make. At this point, standard Monte Carlo Tree Search can be applied.

You can see the huge appeal here for our TCG. The cards themselves would be sub-graphs, with the various rules 
represented in graph form. The legal moves would be represented as edges.

## representing the cards as graphs

