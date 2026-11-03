---
title: "Can AI Playtesting Help Design a Trading Card Game? Part 1: Generating Cards and a GNN Agent"
description: "Generating original cards for a trading card game from a grammar, and training a graph neural network that can play cards it has never seen, so self-play can playtest and price them."
images: ["/images/tcg/tcg_part1_thumbnail.jpg"]
date: 2026-11-03
mathjax: false
tags: ["TCG", "Machine Learning", "ML", "AI", "Graph Neural Networks", "GNNs", "Procedural Generation"]
draft: false
thumbnail: "/images/tcg/tcg_part1_thumbnail.jpg"
---

## Intro

Years ago at Mediatonic we worked on a card game in the mould of Hearthstone, and I watched the designers
struggle with balance. You want cards with interesting synergies, but no card so strong that every deck has to
run it. Every new card interacts with every old one, so checking a set by hand gets harder with each card you
add.

It isn't just us. Hearthstone, one of the best-known digital trading card games, has released thousands of
cards over more than a decade, made by experienced designers. Even so, cards are regularly weakened ("nerfed") after launch, once millions of players find a combination
nobody spotted, and Magic: The Gathering keeps lists of cards banned from its formats for the same reason.

The idea I've had since those Mediatonic days: what if an agent could **playtest** the set for you? Better still, what if it
could take a set that's badly balanced and tune it, playing thousands of games and adjusting costs until the
cards are fair? This series is my attempt to do that for a whole set of around 3,000 cards, for a card game
written from scratch:

- **Part 1** (this post) builds the pieces: a card generator, an agent that can play any card it generates, and
  a simple first pass at balancing.
- **Part 2** balances the set properly, against the best decks rather than the average one, and gives it the
  structure a real set needs: rarities, a mana curve, and aspects with identities of their own.
- **Part 3** puts the tools to work on something genuinely new: not a twist on Hearthstone, but a game with
  ideas of its own, which still keeps the DNA every trading card game shares.

Parts 1 and 2 are the proof of concept. They use rules close to a game many people know, so it's easy to see
whether the tools work. Part 3 is what they're for: designing a card game that doesn't exist yet.

There are two problems to solve:

1. **Card generation.** Something has to come up with the cards: their stats, keywords and abilities.
2. **An agent that can play cards it has never seen.** Training a game-playing agent is expensive. If it had to
   start again every time a card changed, the loop would never finish. So the agent has to read what a card
   *does* and play it sensibly, the first time it sees it.

Put them together and you get a loop:

{{< demo src="/demos/tcg/design_loop.html" title="The design loop" bar="0px" >}}

Each time round, the loop checks the set against a handful of design metrics: is each card worth its cost, do
the cards work together, are there any infinite combos, are any two cards too alike? In part 1 we'll only look
at the first of these, balancing card costs. The rest come in part 2.

Choosing those metrics, and deciding what a healthy set looks like, is where skilled designers come into
their own. The agent can tell you how strong a card is, not whether it's fun, easy to read, or leaves the
opponent something to play against. So this is a playtesting tool for designers, not a replacement for them:
they decide what to measure, and the loop does the thousands of games.

### Why not just ask an LLM?

The obvious shortcut is to ask a large language model for 3,000 cards. It would produce something that *looks*
like a card set, but on its own it's a poor fit for this project, for two reasons. LLMs have read every Magic
and Hearthstone card ever printed, so without constraints and checks for novelty, what comes out tends to drift
towards cards that already exist. I want cards that are original by construction. And an LLM can't tell you
whether a card is balanced: it can guess, but it can't play 10,000 games with it.

So here the cards come from a small hand-written grammar, and the judging is done by an agent that has actually
played them. (An LLM might still be useful later for naming cards, a job it's well suited to.)

## The game

For part 1, the rules follow Hearthstone's broad structure, on purpose. Lots of people know it, so the results
can be compared with a real, well-understood game, and if you've played Hearthstone or Magic you already know
most of this:

- Each player has a **hero with 30 health**. Reduce the enemy hero to 0 and you win.
- You get one more **mana crystal** each turn, up to 10, and spend mana to play cards.
- **Units** go on the board (up to 7) and attack enemy units or the enemy hero. **Actions** have an effect and
  are gone.
- Five **keywords**: Guard (must be attacked first), Elusive, Charge, Duelist and Armor.
- Decks are 30 cards, at most 2 copies of each. The first player starts with 3 cards, the second with 5 and a
  Spare Spore (one extra mana, once).

The theme is mushrooms with jobs: a Porcini Baker, an Enoki Lamplighter, a Ghost Miner. Jobs work like tribes,
and the six aspects (Ember, Bog, Moss, Meadow, Glow and Frost) are habitats. For now they're just colours.

{{< lightbox src="/images/tcg/help_screen.webp" alt="The game's help screen: how to read a card, and the five keywords" >}}

Under the hood, a rules engine in Python owns every rule and all of the game's state; the Unity client in the
screenshots only draws what the server sends it. Cards are pure data, so a newly generated card never needs new
code.

## Generating cards

Every ability is a **tree**. At the root is *when* it happens (a trigger: "when you play this unit", "at the
start of your turn", "whenever an enemy unit is defeated"), or a static rule that's always on. Below that are
the steps: choose a target, deal damage, draw cards, "if it was defeated, …", "you may …, if you do …",
"choose one: …". Each step has its own options, so the tree grows one decision at a time until every branch
ends in something concrete, like "deal 3 damage to chosen enemy unit".

The grammar is small and closed, so every tree it produces is a legal ability that the engine can run and
that turns into card text. Sampling is the easy part, though: most random trees are bad. Each draw goes
through a series of tests:

1. **Grammar:** is it legal at all?
2. **Quality:** is it small enough to read, are its steps connected, is any "choose one" option strictly worse
   than another, and does it narrow its targets so far that it would almost never fire?
3. **Card:** put it on a card with stats and a cost. Does it fit?
4. **Variety:** throw it away if the set already has too many abilities of the same shape.

The tests only look at the tree's shape, never at particular cards. Making 200 cards took 1,201 draws.

Here is one card's path through the grammar. Press play, or drag the slider to scrub through it; *New card*
picks another from the set.

{{< demo src="/demos/tcg/card_generation_visualiser.html" title="Card generation visualiser: one card's path through the grammar" >}}

## Reading cards with a graph neural network

Now for the second problem: an agent that plays cards it has never seen. Most game-playing networks see the
board as a fixed grid or a fixed list of slots. A card game doesn't fit that: the number of cards changes all
the time, and each card's ability is a tree of its own.

A **graph neural network** (GNN) fits much better. The whole game state becomes a graph: heroes, cards in
hand and on the board, and each card's ability tree as a small graph of its own, hanging off the card. Edges
say how things relate: this card belongs to this player, this step comes after that one, this "it" refers to
that choice. The GNN passes messages along the edges, so each node builds up a picture of its neighbourhood.
The key point is that a card is described only by *what it is* (its stats, keywords and ability graph), never
by an ID. A new card is just a new graph built from familiar parts, so the network can read it straight away.

The agent learns by playing against itself, with PPO. The only reward is +1 for a win and −1 for a loss, and
some of its games are against frozen earlier versions of itself and a hand-written player, so it doesn't just
learn to beat its current self. The hand-written player is simple: it plays the most expensive card it can
afford, attacks a unit when it can kill it and survive, and otherwise goes for the hero. On a set of 200 cards
the agent never trained on, it beats that player in 69% of games (275 of 400). Those cards come from
the same grammar as its training cards, though, so this is new combinations of familiar building blocks rather
than brand-new mechanics: a new kind of effect would need the grammar extending and the network training on it.

None of this needs a big machine. Self-play runs at roughly 5–10 games a second on my Mac, across 10 processes;
a training iteration is 64 games and takes about 20 seconds, and the agent in this post had a few hours of
training in all.

The three demos below show what happens inside the network, one stage at a time.

### 1. From a board to a graph

The board is turned into nodes and edges, and each node gets a vector of numbers: a card's cost, power,
health and keywords, an ability step's type and amount. Here we follow one card as it's embedded.

{{< demo src="/demos/tcg/gnn_visualiser_1.html" title="GNN visualiser, part 1: the board as a graph" >}}

### 2. One layer of message passing

Each layer, every node collects messages from its neighbours. The layers are
[GATv2](https://arxiv.org/abs/2105.14491) graph attention layers, with their own weights for each kind of edge:
attention decides how much each neighbour matters, so a unit can listen more closely to the enemy Guard in front of it than to a card in hand. After a
few layers, a card's vector carries what's going on around it.

{{< demo src="/demos/tcg/gnn_visualiser_2.html" title="GNN visualiser, part 2: one GNN layer, step by step" >}}

### 3. Policy and value

Two heads read the final vectors. The **policy** scores every legal action (play this card, attack that unit
with this one) and the **value** estimates who is winning. Because each action is scored from the nodes it
involves, the same network handles any number of actions, and any card.

{{< demo src="/demos/tcg/gnn_visualiser_3.html" title="GNN visualiser, part 3: the policy and value heads" >}}

### How good does the agent need to be?

It doesn't need to be superhuman. To measure a design, it needs to play roughly the way people play. If it
finds a line no human would, or misses one every human would, its numbers say more about the agent than about
the cards. Beating a simple scripted player two games in three is a long way from expert play, but it's
enough to tell a strong card from a weak one.

It also has no belief system. It sees its own hand and the board, but it doesn't try to work out what the
opponent is holding, as a good player does. For measuring whether a card is fairly priced, that's probably
fine: a card that's too cheap is too cheap whoever is holding it. Part 2 will push the limits of this setup,
because finding the strongest decks takes a stronger player.

## Balancing, version one

Each generated card starts with a cost from a rough value table: so many points for power, health, each
keyword and each effect. Then the agent tests the table by playing it against variations of itself:

- a deck against the same deck with its abilities removed (do abilities matter?),
- against the same deck with each ability's points put back into plain stats (are abilities priced fairly?),
- against the same deck with its actions swapped for plain units of the same cost (are actions fair?).

The current balancing loop is deliberately small in scope: it only prices each kind of effect, and making
the set truly balanced will be the main focus of part 2. Even so, it caught a lot. The table charged far too
much for abilities: a deck lost three games in four to its own copy with the abilities traded for stats. Only about half of the abilities ever did anything
when played, because their targets often weren't there. So the agent measured what each kind of effect is
actually worth, the costs were updated, the set regenerated and the agent fine-tuned on the new cards. After a
few rounds, a deck with abilities wins about 55–59% against the same deck with stats instead, and actions are
roughly worth their cost. Each of these comparisons is 800 games.

## The first 200 cards

Here they are: the first 200 cards, generated and costed by the loop above. Filter by kind, or switch to
*Thrown away* to see the draws that failed a test, and why.

{{< demo src="/demos/tcg/card_lab.html" title="Card lab: the first 200 cards" height="85vh" >}}

## What's next

On the whole, the grammar has generated some great abilities. A Porcini Baker that deals 4 damage when it's
defeated, a Buttercup Growth for your Miners, an Oyster Postman that makes your other Postmen cheaper: these
are the kinds of cards a designer might write.

A few patterns don't work as designs, though:

- **Turkey-tail Postman – Morning Rounds:** "At the start of your turn, your other units get +2 health this
  turn." When your turn ends, the extra health goes but the damage stays. A unit that survived its attack only
  thanks to the buff then dies as you end your turn, and since units attack once a turn, the extra time buys it
  nothing. Health that lasts through the opponent's turn would make it a real defensive buff.
- **Actions that summon a single unit,** like **Hayfield Gathering** (2 mana: summon a 1/3 Cap Kid). That's a
  unit card with an extra step, so it might as well be a unit card.

Both are easy to fix in the grammar: health buffs can last through the opponent's turn, and a summoning action
can make two or more units or do something else as well. Neither needs a rule about a particular card.

Some cards are also simply too cheap. A few examples:

- **Mire Gossip** (1 mana): draw 4 cards, or deal 2 damage.
- **Firefly Letters** (1 mana): draw 3 cards, or return every enemy unit to its owner's hand.
- **Ghost Miner – Clocking Off** (1 mana): a 1/1 that draws 2 cards at the end of every turn.
- **Truffle Baker – Chicken Soup** (1 mana): heals your hero for 4 at the start of every turn.

Card draw and healing come out much too cheap, and the reason says something about the method. Each effect's
price is its *average* worth over random decks. In a random deck, many of those extra cards are mediocre, so
drawing them isn't worth much. But nobody plays random decks. Players build the **best** deck they can, and in
the best deck an extra card is worth a lot. A card's fair cost depends on the strongest deck it can appear in,
not the average one. One combo that breaks 1% of decks is all anyone will play.

That's part 2: searching for the best decks and balancing the set against them, and it's the exciting bit. A card's
worth depends on the cards around it: jobs that reward each other, triggers that feed each other, combos nobody
designed. The number of possible 30-card decks is astronomical, so finding the strongest ones, and the cards
that make them too strong, is a hard search problem, and it may well take some interesting ML to solve. The grammar
will grow too, with the fixes above and new building blocks for the cards to use.

*AI was used to help build this project, write this post and generate the game's assets.*
