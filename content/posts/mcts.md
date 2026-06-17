---
title: "Interactive Monte Carlo Tree Search Demo"
date: 2026-06-17
mathjax: true
tags: ["MCTS", "Machine Learning", "ML", "AI", "Connect 4"]
draft: true
---

Monte Carlo Tree search (MCTS) is the famous tree-search planning algorithm used Google DeepMind's AlphaGo as well as many other
applications. The algorithm itself is relatively simple and should be seen maybe more as a planning algorithm than a learning one.

I always believe the best way to learn about something is via a hands-on demo, so check out the demo here. Play on the board in
the top right and see how the predicted winner changes as the game progresses. The tree evolves to show which paths work out best for
which player. The circle with the highlight indicates the best move. Be sure to zoom in/out to see the full details lower down the tree.

```
{{</* unity-webgl
  buildPath="/games/mcts_c4/Build"
  buildName="MCTS"
  title="Connect 4 MCTS demo"
  width="960"
  height="600"
*/>}}
```

Now that you've has a play let's talk about how MCTS works. It's probably easiest to start explaining this algorithm once it's a few iterations in.
We start at the root node (the current board state) and traverse the tree selecting the maximum UCT score at each node (we'll explain UCT in a bit).
When we get to a leaf node we then do a rollout. This consists of playing several simulated games from this node by playing random moves, 
giving an indication of the likely result. Once this is complete the result (which will be something like a 0.8 win rate for red) is propagated
back up the tree. Each node tracks it's visit count and sums the win rates from the rollout.

Now let's take a deeper look at the UCT (Upper Confidence Bound) score, given by the following formula:

$$UCT = \frac{WinCount}{VisitCount} + C\sqrt{\frac{\log{(VisitCount_{parent})}}{VisitCount}}$$

Let's break this down:

The first term is the win rate for this node. A higher win rate suggests a better node/action. This is also 
referred to as the estimated value of this node. Note that because of how the rollout works $WinCount$ might not be an integer. 

The second term is inversely related to the number of times this node has been visited relative to its sibling nodes i.e. low visit 
counts relative to its siblings will give a higher score. The term in the denominator here $VisitCount_{parent}$ can 
initially appear confusing as it's referring to the parent node and not the sibling nodes, but because of the backpropagation 
step the parent visit count is equal to the sum of the child visit counts.

The purpose of this second term is to balance **exploitation** vs **exploration**. If we only used the value term here we might get a 
situation where one child node has a win rate of, say, 0.6 with only one visit and a sibling node has a win rate of 0.61 after many visits.
It stands to reason that the node with only one visit that has only a marginally lower value score warrants further exploration. The constant $C$
tunes this balance between exploration and exploitation, and is typically set to around $1.4$

Note that you'll often see the UCT score expressed in the more concise form:

$$UCT = V_i + c\sqrt{\frac{\ln{N}}{n_i}}$$
