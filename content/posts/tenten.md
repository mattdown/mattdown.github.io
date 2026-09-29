---
title: "Using an AI Agent to explore game dynamics in 1010!"
date: 2026-09-28
mathjax: true
tags: ["Machine Learning", "ML", "Reinforcement Learning", "RL", "AI", "Game Design", "Game Prototyping"]
draft: false
thumbnail: "/images/tenten/tenten_thumbnail.png"
---

## Intro

For a while now I've been interested in **AI accelerated design space exploration**. It's a compelling approach that 
allows the rapid iteration of game rules and balance. As someone who's spent a lot of their career working on 
early stage prototypes, I've seen firsthand how important it is to nail the core mechanics of a game.

So how does that work? You first need to train an agent that is capable of playing the game in a way that it 
can react to balance changes and even rule changes. Then you define some metrics based on game dynamics that I'll 
cover in the next section, before changing the game balance and rules to hit those metrics.

{{< lightbox src="/images/tenten/design_loop.png" alt="The design loop: change the rules, playtest with an AI agent, measure design metrics, then tune" >}}

This isn't a new idea. Researchers have been using AI players to explore game designs since at least the late 2000s,
and I've listed some of the work I've found most interesting in [Related work](#related-work) at the end. What's
changed is how accessible it's become: the agent in this post trained in about 40 minutes on a laptop.

As an initial proof of concept, I'm going to look at the already highly successful puzzle game **1010!** and show how 
we can train an agent to play the game and uncover some of the reasons the game dynamics work so well in this game. 
If you've not played 1010! before, feel free to skip to the [results section](#results) where you can have a quick play.

### Design principles

Let's now look at two design principles that are interesting to explore in 1010!, and that we can easily build
metrics for.

But before we do that, I think there's 
an important caveat to point out. Choosing which principles (and metrics) to focus on is part of the creative process. 
Designers will naturally want to focus on different things, leading to games that are fun in 
different ways. This isn't a case of AI taking over and deciding what is/isn't fun.

#### Complexity of decision-making

During a typical game is there an overwhelming number of plausible moves that can't easily be narrowed down to one 
or two good ones? Or maybe there is a uniform distribution of plausible moves, which makes the player feel it 
doesn't matter what they choose. How does this distribution change over time? Does it start with a small number of possibilities 
and then balloon rapidly as the game evolves, leaving the player feeling overwhelmed?

#### Inevitability of outcome

Once you're part-way through a typical game, how inevitable is the survival outcome based on your play so far? 
e.g. does a mistake on turn 2 massively impact your chance of survival 100 moves later, or does it only 
impact your survival chances over the next 10 moves? The latter feels much better to players as it makes them feel they 
still have something to continue to play for even if things aren't looking great.

## Methodology

Here's the setup. I've kept this section fairly light, the full details are in the [Appendix](#appendix) for anyone
who wants them.

### The game

I implemented a 1010!-style block puzzle in Python. The rules are:

- A 10×10 grid that starts empty.
- You're dealt a hand of **3 random pieces**. You have to place all 3, in any order, before you get the next 3.
- There are 16 piece shapes (lines, squares and L shapes) and they can't be rotated.
- Any completely filled row or column is cleared.
- It's game over when none of the pieces left in your hand fit anywhere.

For scoring I've used **turns survived**, i.e. the number of pieces placed. The real game awards points for line
clears, but survival is what actually matters and it keeps the analysis simple.

The 3-piece hand turns out to be quite important. It means every game runs in exact 3-move cycles, and you'll see
that rhythm show up all over the results.

### Game agent

The agent is about as simple as it gets. It's a small convolutional neural network (CNN) that looks at a position
(the board plus the pieces still in hand) and outputs a single number, the **value** of that position. This is
roughly "how many more turns do I expect to survive from here?".

{{< lightbox src="/images/tenten/board_evaluator.png" alt="Architecture of the board evaluator network" >}}

To pick a move, the agent tries every legal placement, asks the network to rate each resulting position and plays the
highest-rated one. There's no look-ahead or tree search, it's purely greedy. The network is tiny, only ~79k
parameters.

For training, a hand-written heuristic agent (which likes clearing lines and dislikes leaving holes) played the first
batch of games. The network learned from those, overtook the heuristic after a single round, and from then on learned
from its own games (self-play). The whole thing took ~37 minutes on a laptop.

### Unity - Python pipeline

{{< lightbox src="/images/tenten/how_the_game_runs.png" alt="How the game runs: Unity talking to a Python server in development, and fully in the browser for the WebGL build" >}}

During development, all the game logic lives in Python. The Unity client only handles the board, drag and drop and
animations. It sends each move to a Python game server as JSON over ZeroMQ, and the server checks the move, applies it
and sends the new board back. That way there's a single copy of the rules, shared with training and the analysis.

The big advantage of setting it up like this is that the game lives right next to the machine learning code, so the
model can be trained directly with PyTorch. Training plays its games in Python without Unity in the loop, so it can
get through thousands of them quickly.

The web version you can play below has no Python server to talk to, so the rules were ported to C# and the network
exported to ONNX, which runs in the browser with Unity's Inference Engine. The client has a local mode that gives the
same responses as the Python server, so the rest of the game doesn't know the difference. To make sure nothing got
lost in translation, the C# port and the exported model were checked against real positions exported from the Python
game.

### Calculating and defining metrics

Now to turn the two design principles from earlier into hard metrics that can be measured.

**Complexity of decision-making → entropy**

The natural tool here is [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)), measured in
bits. If you have $N$ equally good options, the entropy is $\log_2 N$ bits, so it's easy to turn back into a count:
$2^H$ is the "effective number of options". 4 bits is like choosing between 16 equally good moves, 0 bits means
there's only one sensible move.

I measure two versions of this at every turn:

1. **Legal-move entropy**, $H = \log_2 N$ where $N$ is the number of legal moves. This is how much freedom the board
   gives you.
2. **Agent entropy.** The network rates every legal move, and I turn those ratings $v_i$ into probabilities with a
   softmax, then take the entropy:

$$p_i = \frac{e^{v_i/T}}{\sum_j e^{v_j/T}}, \qquad H = -\sum_i p_i \log_2 p_i$$

If many moves are rated about the same, the entropy is high and the decision is genuinely hard. If one move clearly
stands out, it's close to 0. This is the one that maps onto "plausible moves that can't easily be narrowed down".
Note the agent always plays its top move, the softmax is purely a measuring tool. ($T$ is a temperature setting, see
the appendix.)

**Inevitability of outcome → does a bad position stay bad?**

Here I use the network's value directly. If the outcome is inevitable, a bad position now should mean a bad
position later, and players who fall behind should rarely recover. So I look at:

- **Autocorrelation of the value:** if a position is rated above average now, is it still above average $n$ moves
  later? This tells us how long the network's rating "remembers" anything.
- **Dips and recoveries:** I call it a **dip** when the value falls into the bottom 10% of mid-game values, and a
  **recovery** when it climbs back to the typical (median) value before game over. What fraction of dips recover,
  and how quickly?

## Results

Here are the results. But first, here's the Unity game with the agent offering hints to give you an idea of how it
plays.

{{< unity-webgl
  buildPath="/games/PyTenTen/Build"
  buildName="PyTenTen"
  fileSuffix=".unityweb"
  title="1010! with AI hints"
  width="2000"
  height="2000"
>}}

All the results below come from 1000 games played by the agent, 300,889 moves in total.

### Agent training

Here's how the trained agent compares with the heuristic and with random play:

| Agent | Games | Mean turns survived | Median | Best |
|---|---|---|---|---|
| Random | 500 | 17 | 17 | 37 |
| Heuristic | 200 | 118 | 97 | 374 |
| **Neural network** | 1000 | **301** | **220** | **1811** |

So the network survives ~2.5× longer than the heuristic and ~18× longer than random play. That's pretty good for a
network this small with no look-ahead.

{{< lightbox src="/images/tenten/score_distribution.png" alt="Histogram of turns survived over 1000 games" >}}

The spread of scores is interesting in its own right. It's heavily skewed: most games end within a few hundred
turns, but there's a long tail of games going past 1000. It fits a log-normal distribution well, which is what you'd
expect if survival builds up from lots of small, independent risks. Every new hand is another chance to be dealt
something that doesn't fit. Keep that in mind, luck comes back later.

### Complexity of decision-making

{{< lightbox src="/images/tenten/action_entropy.png" alt="Legal-move and agent entropy over the course of a game" >}}

The top row shows both entropies over the course of a game (blue for legal moves, orange for the agent). On the left
the games are lined up by when they ended, on the right by when they started. Lines are the median across games, the
bands cover the middle 50%.

A few things stand out:

- **There's a lot of choice, but it can be narrowed down.** A typical board allows ~44 legal moves (5.4 bits), but the
  agent only treats ~16 of them as serious candidates (4.0 bits). That feels like a good place for a puzzle game to
  sit. It's far from obvious, but it's not an overwhelming Go-style wall of options either. And
  it's not the case that every move is as good as any other.
- **It's remarkably flat.** After the opening (the empty board allows loads of moves), both entropies settle within
  ~30–40 turns and then stay flat for hundreds of moves. The board doesn't slowly fill up over time.
- **Then it collapses.** Over the last ~20–30 moves both drop sharply. The agent's entropy starts falling slightly
  before the legal-move count, which fits running out of *good* moves before you run out of *legal* ones.

The medians hide a lot of move-to-move variety though. Here are the first 100 moves of 10 individual games, with each
hand of 3 shaded:

{{< lightbox src="/images/tenten/action_entropy_traces.png" alt="Entropy traces for 10 individual games" >}}

The legal moves (blue) follow a very regular sawtooth: the most freedom on the first piece of each hand, the least on
the last. The agent (orange) is more interesting. There are one-off spikes down to ~0 bits where one move is clearly
best (probably a line clear on offer), and longer troughs lasting several hands when the board gets tight, e.g. game
39 around turns 25–60.

{{< lightbox src="/images/tenten/action_entropy_hand_cycle.png" alt="Entropy by position in the 3-piece hand" >}}

Splitting this up by position in the hand makes the rhythm clear:

| | 1st piece (3 in hand) | 2nd piece (2 in hand) | 3rd piece (1 in hand) |
|---|---|---|---|
| Legal moves | 6.19 bits | 5.73 bits | 4.75 bits |
| Agent | 4.01 bits | 4.50 bits | 3.96 bits |

The board's freedom pulses strongly with the hand, but the agent's decisions barely do. Having 3 pieces gives you
more placements, but it doesn't make the decision harder. My guess (untested) is that with 3 pieces the playing order
matters a lot, e.g. setting up a line clear, which makes one move stand out.

### Inevitability of outcome

{{< lightbox src="/images/tenten/value_autocorr.png" alt="Autocorrelation of the network's value, and value in the last 100 moves" >}}

The left panel is the autocorrelation of the network's value. It drops below 1/e (a common "it's mostly forgotten"
threshold) after **8 moves**, and is essentially gone by **~19 moves**. In other words, knowing a position is good or
bad right now tells you very little about how things will look 6 hands later.

The right panel shows the median value in the run-up to game over. It's flat until ~20–30 moves before the end and
only then starts to fall. The network rates a game with 50 moves left the same as one with 500. As far as the
network can tell, beyond ~20 moves the outcome comes down to pieces that haven't been dealt yet.

Now for my favourite chart:

{{< lightbox src="/images/tenten/value_recovery.png" alt="Chance of recovering from a bad position, and outcomes of dips" >}}

On the left is the chance that a position gets back to a typical value before game over, plotted against how bad
it is right now. On the right, for all 4771 dips across the 1000 games, is the share that have recovered (blue) or hit
game over (orange) as the moves go by.

- **81% of dips recover**, with a median of 9 moves.
- **19% end in game over**, with a median of 8 moves.
- Almost every dip is settled one way or the other within ~30 moves.
- **There's no point of no return.** The chance of recovery falls smoothly as the position gets worse, and even the
  worst 1% of positions recover 44% of the time.

So you can usually get out of a bad position quickly, unless you can't, and then it's over just as fast.

You can see this in individual games too. Here are the last 100 moves of the same 10 games as before, with the
typical value (dashed) and the dip threshold (dotted):

{{< lightbox src="/images/tenten/value_traces_end.png" alt="Network value over the last 100 moves of 10 games" >}}

The endings come in roughly three flavours:

- **A steady slide** over the last ~15 moves (games 70, 816, 845 and 623).
- **Sudden death.** Game 170 recovers from a dip 20 moves from the end, climbs back above typical, then dies within
  the last 2 moves.
- **Hovering around the threshold** for the last 20–30 moves before finally going under (games 12 and 39).

### What do these metrics tell us about 1010!?

On **complexity of decision-making**, 1010! sits in a sweet spot. There are always plenty of legal moves, but the agent can narrow them down
to a manageable shortlist of ~16. That stays consistent for the whole game, until the final crunch.

On **inevitability of outcome**, 1010! is very much *not* inevitable, and I think that's a big part of why it works.
Being halfway through a game tells you almost nothing about how it'll end. A bad position is usually recoverable,
and even a really bad one gives you a fighting chance. But recovery isn't guaranteed either, so there's always some
tension. As a caveat, all of this is measured through the network's opinion, which has its own blind spots (more on
this in the appendix).

When designing new casual puzzle games these can be used to guide your own game dynamics, both in terms of the 
design principles, but also the exact numbers that 1010! hits.

That's only half of the loop from the intro though. In a follow-up post I'll close it by changing the rules and
balance of 1010! and seeing how these metrics move.

## Related work

Using AI players to explore game designs has a surprisingly long history. Here's some of the work closest to this
post, if you want to dig deeper:

- **Measuring what makes a game good.** Cameron Browne's [Ludi system](https://cambolbro.com/cv/publications/ciaig-browne-maire-19.pdf)
  (*Evolutionary Game Design*) evolved new board games and scored them through self-play against 57 aesthetic
  criteria. These include *drama* (players should have hope of recovering from bad positions) and *uncertainty* (the
  outcome should remain uncertain for as long as possible), which are very close to the inevitability metric here.
  One of its games, Yavalath, became the first commercially released board game designed entirely by a machine.
- **Exploring a game's parameter space.** Isaksen, Gopstein and Nealen's
  [Exploring Game Space Using Survival Analysis](http://www.nealen.net/papers/exploring-game-space-FDG2015.pdf) used an
  AI player to explore thousands of variants of Flappy Bird, predicting each one's difficulty from the distribution
  of scores.
- **Changing the rules with a superhuman agent.** DeepMind's
  [Assessing Game Balance with AlphaZero](https://arxiv.org/abs/2009.04374), with former world champion Vladimir
  Kramnik, trained AlphaZero on nine rule variants of chess and compared how decisive each one was. This is the full
  design loop from the intro, just with a vastly bigger agent.
- **Mobile puzzle games.** Roohi et al.'s
  [Predicting Game Difficulty and Churn Without Players](https://arxiv.org/abs/2008.12937) combined deep
  reinforcement learning agents with a simulated player population to predict pass rates and churn for each level of
  Angry Birds Dream Blast.
- **Letting AI change the rules too.** More recently, language models have started to take on the designer's side of
  the loop. [GAVEL](https://arxiv.org/abs/2407.09388) uses a language model to mutate and recombine board games
  written as code, and [RuleSmith](https://arxiv.org/abs/2602.06232) uses LLM agents playing each other, plus
  Bayesian optimisation, to tune the rules of a civilization-style game for balance.

For a broader overview, [AI for Games in the Foundation Model Era](https://arxiv.org/abs/2609.16679) is a recent
survey of how large AI models are being used across game development, from playing and design through to testing.

## Appendix

### A. Game details

- **Pieces:** 16 shapes, drawn uniformly at random with no rotation:
  - lines of 3, 4 and 5 cells, horizontal and vertical (6 shapes);
  - 2×2 and 3×3 squares;
  - small (3-cell) and large (5-cell) L shapes, 4 rotations each.
- **Clearing:** several rows and columns can clear at once.
- **Reward:** +1 per move survived. Points for line clears exist in the code but are switched off.
- **Action space:** 3 pieces × 14 × 14 anchor positions = 588 possible actions, most of which are illegal at any
  given moment. The anchor is the top-left of the piece's 5×5 bounding box, from −4 to 9 on each axis.

### B. The agents

| Agent | How it picks a move |
|---|---|
| Random | A uniformly random legal move. |
| Heuristic | Tries every legal move and scores the result: +1000 per line cleared, −10 per hole (an empty cell boxed in on all 4 sides), −1 per edge between filled and empty cells (roughness). |
| Neural network | Tries every legal move and asks the network to rate each resulting position (board + remaining pieces). Plays the highest-rated one. |

### C. Network architecture

- **Input:** 4 channels on a 10×10 grid.
  - Channel 0: the board (1 = filled).
  - Channels 1–3: one per piece in hand, with its 5×5 layout in the top-left corner and the rest zero. A piece that
    has already been played is all zeros.
- **Trunk:** 3× Conv 3×3 (64 channels, padding 1) + ReLU. There's no pooling, so the 10×10 grid is kept the whole
  way through.
- **Value head:** Conv 1×1 (32 channels) + ReLU → global average pooling → fully connected 32 + ReLU → fully
  connected 1.
- **Size:** 79,393 parameters.

**What the value means.** The network is trained to predict the discounted number of future turns survived,
$\sum_k 0.98^k$ over the remaining moves, scaled to [0, 1]. A discount of 0.98 gives an effective horizon of ~50
moves, so positions 100 or 500 moves from death have almost the same target. The network is never asked to tell
them apart.

### D. Training

{{< lightbox src="/images/tenten/training_and_analysis.png" alt="How the AI was trained and analysed" >}}

- **Data generation:** each generation plays 1000 games with a "teacher" agent, which makes a random move 50% of the
  time for variety. Every position is recorded with its discounted return, computed backwards from game over (+1
  per move, discount 0.98). The targets are scaled by the largest return seen.
- **Augmentation:** each position is also added in all 8 rotations and reflections. The board and the pieces are
  transformed together, so the pieces still fit.
- **Optimisation:** mean squared error, Adam (learning rate 0.01, weight decay 1e-4), batch size 8000, 20 epochs per
  generation. The learning rate is halved when validation loss stalls (patience 3), with an 80/20 train/validation
  split. The best validation checkpoint within each generation is kept.
- **Curriculum:** the heuristic is the teacher at first. Once the network out-survives the heuristic by 5% (over 50
  games each), it switches to self-play. 10 generations in total.
- **How it went:** after generation 0 (heuristic data only) the network survived 26 turns, against the heuristic's
  120. After generation 1 it survived 228, beat the heuristic and switched to self-play for generations 2–9. Final
  validation loss was 0.0239, and the whole run took ~37 minutes on an Apple Silicon GPU.

### E. Metric details

- **Temperature.** All agent-entropy charts use $T = 0.01$. The network's scores are in [0, 1] and the gap between the
  best and the median move is only ~0.01–0.07, so at $T = 1$ every move would look equally likely. $T$ is a free
  parameter. The patterns hold across temperatures, but the absolute numbers depend on it.
- **Normalised entropy:** $H / \log_2 N$, from 0 to 1. This is the agent's decisiveness *relative to* how many options
  it has (the bottom row of the entropy chart). It sits at ~0.8 for most of the game and dips to ~0.72 at the end.
- **Mid-game:** turn 30 onwards, excluding the last 30 moves. Used for the steady-state analyses, because entropy and
  value are still settling after the empty board early on, and collapse at the end.
- **Autocorrelation at lag $n$:** the correlation between a quantity at turn $t$ and at turn $t + n$, using each game's
  differences from its own mid-game average, averaged across games (only games long enough for the lags shown).
- **Hand cycle removed:** before correlating, subtract each game's average for each position in the hand (1st, 2nd,
  3rd piece). This leaves only structure slower than the 3-move rhythm.
- **Dips:** the typical value is the median mid-game value, 0.348. A dip starts when the value first falls into the
  bottom 10% of mid-game values (below 0.274), and a new dip can only start after the previous one has recovered.
  926 of the 1000 games ended in a dip that never recovered. The other ~7% died without ever reaching the bottom 10%,
  a sudden death from a bad deal.

### F. Extra charts

**Entropy autocorrelation.** The same autocorrelation idea applied to entropy, raw (left) and with the hand cycle
removed (right):

{{< lightbox src="/images/tenten/action_entropy_autocorr.png" alt="Autocorrelation of legal-move and agent entropy" >}}

The legal-move entropy has sharp peaks every 3 moves that barely fade, because the game deals a new hand every 3
moves regardless. With the cycle removed, it decays smoothly to ~0 by ~20 moves and never goes negative, so there's
no longer hidden cycle, just persistence. A tight board stays tight for a couple of hands, then drifts back. The
agent's entropy decorrelates much faster: its decisiveness is mostly move-to-move.

**Value in the opening.** The first 100 moves of the same 10 games:

{{< lightbox src="/images/tenten/value_traces.png" alt="Network value over the first 100 moves of 10 games" >}}

The value starts at ~0.48 on the empty board and settles within ~30 turns. Game 39 spends turns ~22–55 at or below the
dip threshold, the same stretch as its entropy trough, then climbs back above typical. Game 816 dips deep (~0.18)
around turns 70–95 and recovers.

### G. Limitations and next steps

- **The value is the network's opinion, not the truth.** How short-sighted it is partly reflects how it was trained:
  a 0.98 discount (a ~50-move horizon) and targets taken from games with 50% random moves, not from its own greedy
  play. So "how inevitable the outcome is" is measured *through* the network. A more direct test is to replay the
  same position many times with different deals and measure how widely the outcomes spread. That's a good experiment
  for the follow-up post.
- **The 3rd-piece blind spot.** When one piece is left, the network scores each candidate position with all three
  piece slots empty, which never occurs in training. The last piece of each hand is therefore judged partly outside
  what the network has seen. This is a known issue with a straightforward fix: include those positions in the
  training data and retrain.

---

*AI disclaimer: AI tools were used to help with the coding, to build the demo and to write this blog post. The ideas
and questions being investigated are my own.*
