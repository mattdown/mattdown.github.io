---
title: "Interactive Monte Carlo Tree Search Demo"
date: 2026-06-17
mathjax: true
tags: ["MCTS", "Machine Learning", "ML", "AI", "Connect 4"]
draft: true
---
# Hands on demonstration of Monte Carlo Tree Search 

Monte Carlo Tree search (MCTS) is the famous tree-search planning algorithm used in Google DeepMind's AlphaGo as well as many other
applications. The algorithm itself is relatively simple and should be seen maybe more as a planning algorithm than a learning one.

At a high level how does MCTS work? Instead of exhaustively searching all possibilities, MCTS searches the tree in a 
principled way that prioritises exploration of the most promising paths within that tree.

I always believe the best way to learn about something is via a hands-on demo, so check out the demo here. Play on the board in
the top right and see how the predicted winner changes as the game progresses. The tree evolves to show which paths work out best for
which player. The circle with the highlight indicates the best move. Be sure to zoom in/out to see the full details lower down the tree.

## Demo

{{< unity-webgl
  buildPath="/games/mcts_c4/Build"
  buildName="mcts_c4"
  fileSuffix=".unityweb"
  title="Connect 4 MCTS demo"
  width="960"
  height="600"
>}}

## Algorithm breakdown

Now that you've has a play let's talk about how MCTS works. It's probably easiest to start explaining this algorithm once it's a few iterations in.
We start at the root node (the current board state) and traverse the tree selecting the maximum UCT score at each node (we'll explain UCT in a bit).
When we get to an unvisited node we add that to the tree and do a rollout/simulation from this new node. This consists of playing several simulated games from this node by playing random moves, 
giving an indication of the likely result. Once this is complete the result (which will be something like a 0.8 win rate for red) and the visit count is propagated
back up the tree. Each node tracks it's visit count and sums the win rates from the rollout.

### UCT score

Now let's take a deeper look at the UCT (Upper Confidence bound 1 applied to Trees) score, given by the following formula:

$$UCT = \frac{WinCount}{VisitCount} + C\sqrt{\frac{\log{(VisitCount_{parent})}}{VisitCount}}$$

Let's break this down:

The first term is the win rate for this node. A higher win rate suggests a better node/action. This is also 
referred to as the estimated **value** of this node. Note that because of how the rollout works $WinCount$ might not be an integer. 

The second term is inversely related to the number of times this node has been visited relative to its sibling nodes i.e. low visit 
counts relative to its siblings will give a higher score. The term in the numerator here $VisitCount_{parent}$ can 
initially appear confusing as it's referring to the parent node and not the sibling nodes. Because of the backpropagation 
step the parent visit count is equal to the sum of the child visit counts and saves us having to sum these each time.

The purpose of this second term is to balance **exploitation** vs **exploration**. If we only used the value term here we might get a 
situation where one child node has a win rate of, say, 0.6 with only one visit and a sibling node has a win rate of 0.61 after many visits.
It stands to reason that the node with only one visit that has only a marginally lower value score warrants further exploration. The constant $C$
tunes this balance between exploration and exploitation, and is typically set to around $1.4$

Note that you'll often see the UCT score expressed in the more concise form:

$$UCT = V_i + C\sqrt{\frac{\ln{N}}{n_i}}$$

### Action selection

Once the tree has been expanded up to a given threshold it's time for the agent to select the best action. Here the UCT score is *not* used. Instead the child with the 
highest visit count is selected. This is because the exploration term in UCT is no longer needed, no time to explore - time to take action now. Simply taking the action
with the highest win rate isn't desirable either as this might select a node with a high value score but low visit count. Total visit count is a better proxy for the child node
with the highest true value.

## Final remarks

Feel free to check out the demo again now that we've gone over the algorithm in more detail. The color of each line/node indicates the value component of the UCT score (win rate) 
and the size of the circle indicate the visit count relative to its siblings. Note that these are the parameters used: rollout/simulation count: $5$, expansion count: $10000$
C: $1.4$. Even running on relatively low-end hardware the algorithm plays a mean (but not unbeatable) game of Connect 4 in realtime. Well done if you manage to beat it without the hints.

As a final note as to how AlphaGo/AlphaZero use MCTS - instead of exclusively relying on random rollouts to assess the value of a given node, and UCT to select which child nodes to explore. 
AlphaGo/AlphaZero uses MCTS in combination with deep neural networks. One netowrk to act as a value network (better than random rollouts) and another as a policy network (better than a simple
UCT score). As the model progresses they use these networks alone rather than MCTS. As you can imagine the random rollout for moves early in a 19x19 game of Go will be particularly poor at
predicting a winner, hence the need for deep neural networks.
