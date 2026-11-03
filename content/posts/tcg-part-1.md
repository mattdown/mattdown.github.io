---
title: "Can AI Design a Trading Card Game from Scratch? Part 1: Generating the Cards"
description: "Generating an original trading card game's cards from a grammar, and training a graph neural network that can play cards it has never seen, so self-play can playtest and price them."
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

The idea I've had ever since: what if an agent could **playtest** the set for you? Better still, what if it
could take a set that's badly balanced and tune it, playing thousands of games and adjusting costs until the
cards are fair? This series is my attempt to do that for a whole set of around 3,000 cards, for a card game
written from scratch.

There are two problems to solve:

1. **Card generation.** Something has to come up with the cards: their stats, keywords and abilities.
2. **An agent that can play cards it has never seen.** Training a game-playing agent is expensive. If it had to
   start again every time a card changed, the loop would never finish. So the agent has to read what a card
   *does* and play it sensibly, the first time it sees it.

Put them together and you get a loop:

{{< demo src="/demos/tcg/design_loop.html" title="The design loop" bar="0px" >}}

### Why not just ask an LLM?

The obvious shortcut is to ask a large language model for 3,000 cards. It would produce something that *looks*
like a card set, but it's a poor fit for this project for two reasons. LLMs have read every Magic and
Hearthstone card ever printed, so what comes out tends to be a remix of existing cards, sometimes copied nearly
word for word. I want a set that is original by construction. And an LLM can't tell you whether a card is
balanced: it can guess, but it can't play 10,000 games with it.

So here the cards come from a small hand-written grammar, and the judging is done by an agent that has actually
played them. (An LLM might still be useful later for naming cards. That's the one job it's well suited to.)

## The game

The rules follow Hearthstone's broad structure, on purpose. Lots of people know it, so the results can be
compared with a real, well-understood game, and if you've played Hearthstone or Magic you already know most of
this:

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

The agent learns by playing against itself (PPO self-play, with past versions of itself mixed in). On cards
it has never trained on, it now beats a hand-written heuristic player about 75% of the time.

The three demos below show what happens inside the network, one stage at a time.

### 1. From a board to a graph

The board is turned into nodes and edges, and each node gets a vector of numbers: a card's cost, power,
health and keywords, an ability step's type and amount. Here we follow one card as it's embedded.

{{< demo src="/demos/tcg/gnn_visualiser_1.html" title="GNN visualiser, part 1: the board as a graph" >}}

### 2. One layer of message passing

Each layer, every node collects messages from its neighbours. Attention decides how much each neighbour
matters, so a unit can listen more closely to the enemy Guard in front of it than to a card in hand. After a
few layers, a card's vector carries what's going on around it.

{{< demo src="/demos/tcg/gnn_visualiser_2.html" title="GNN visualiser, part 2: one GNN layer, step by step" >}}

### 3. Policy and value

Two heads read the final vectors. The **policy** scores every legal action (play this card, attack that unit
with this one) and the **value** estimates who is winning. Because each action is scored from the nodes it
involves, the same network handles any number of actions, and any card.

{{< demo src="/demos/tcg/gnn_visualiser_3.html" title="GNN visualiser, part 3: the policy and value heads" >}}

## Balancing, version one

Each generated card starts with a cost from a rough value table: so many points for power, health, each
keyword and each effect. Then the agent tests the table by playing it against variations of itself:

- a deck against the same deck with its abilities removed (do abilities matter?),
- against the same deck with each ability's points put back into plain stats (are abilities priced fairly?),
- against the same deck with its actions swapped for plain units of the same cost (are actions fair?).

The current balancing loop is deliberately small in scope: it only prices each kind of effect, and making
the set truly balanced will be the main focus of part 2. Even so, it caught a lot. The table charged far too
much for abilities: a deck lost three games in
four to its own copy with the abilities traded for stats. Only about half of the abilities ever did anything
when played, because their targets often weren't there. So the agent measured what each kind of effect is
actually worth, the costs were updated, the set regenerated and the agent fine-tuned on the new cards. After a
few rounds, a deck with abilities wins about 55–59% against the same deck with stats instead, and actions are
roughly worth their cost.

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
  turn." Health that disappears at the end of the turn is hard to follow, and it rarely matters. A short-lived
  power boost makes sense; short-lived health mostly doesn't.
- **Actions that summon a single unit,** like **Hayfield Gathering** (2 mana: summon a 1/3 Cap Kid). That's a
  unit card with an extra step, so it might as well be a unit card.

Both are easy to fix in the grammar: health buffs can last, and a summoning action can make two or more units or
do something else as well. Neither needs a rule about a particular card.

Some cards are also simply too cheap:

- **Mire Gossip** (1 mana): draw 4 cards, or deal 2 damage.
- **Firefly Letters** (1 mana): draw 3 cards, or return every enemy unit to its owner's hand.
- **Ghost Miner – Clocking Off** (1 mana): a 1/1 that draws 2 cards at the end of every turn.
- **Truffle Baker – Chicken Soup** (1 mana): heals your hero for 4 at the start of every turn.

<!-- TODO: check these against the card lab before publishing; swap in any you'd rather show -->

Card draw and healing come out much too cheap, and the reason says something about the method. Each effect's
price is its *average* worth over random decks. In a random deck, many of those extra cards are mediocre, so
drawing them isn't worth much. But nobody plays random decks. Players build the **best** deck they can, and in
the best deck an extra card is worth a lot. A card's fair cost depends on the strongest deck it can appear in,
not the average one. One combo that breaks 1% of decks is all anyone will play.

That's part 2: searching for the best decks and balancing the set against them. The grammar will grow too, with
the fixes above and new building blocks for the cards to use.
