---
title: "Using AI Playtesting to Design an Original Trading Card Game, Part 1: Generating Cards and a GNN Agent"
description: "Generating original cards for a trading card game from a grammar, and training a graph neural network that can play cards it has never seen, so self-play can playtest them."
images: ["/images/tcg/tcg_part1_thumbnail.jpg"]
date: 2026-11-03
mathjax: false
tags: ["TCG", "Machine Learning", "ML", "AI", "Graph Neural Networks", "GNNs", "Procedural Generation"]
draft: false
thumbnail: "/images/tcg/tcg_part1_thumbnail.jpg"
series: "Using AI Playtesting to Design an Original Trading Card Game"
repository: "https://github.com/mattdown/TCG-Project"
---

Years ago at Mediatonic, balancing a Hearthstone-style card game was one of our biggest design challenges. Every
new card interacts with every old one, so checking a set by hand gets harder with each card you add. Even Hearthstone
and Magic: The Gathering, with experienced teams, weaken or ban cards after launch.

What if an agent could **playtest** the set for you, playing thousands of games and helping designers tune it?
This series tries that for a set of around 3,000 cards, for a game written from scratch:

- **Part 1** (this post): a card generator, and an agent that can play cards it has never seen.
- **Part 2:** balancing the set against the best decks, not the average one.
- **Part 3:** using the tools to make an original game that stands on its own.

The post is mostly interactive demos. Each one stands alone, so skip to whichever interests you. The details
are in the [GitHub repository](https://github.com/mattdown/TCG-Project).

## The loop

Generate cards, let the agent play them, measure, tune, repeat. As with all AI playtesting, the design metrics
are where human creativity comes in: designers choose what to measure and what to change. The AI doesn't take
over the key design decisions. It shows their impact quickly, without lengthy rounds of human playtesting.

{{< demo src="/demos/tcg/design_loop.html" title="The design loop" bar="0px" >}}

## The game

For now the rules follow Hearthstone, so the results are easy to compare with a game people know. The theme is
mushrooms with jobs.

{{< lightbox src="/images/tcg/help_screen.webp" alt="The game's help screen: how to read a card, and the five keywords" >}}

## Generating cards

Every ability is a tree, grown one decision at a time from a small grammar, then tested: is it readable, does
it fit on a card, is it too like the others? Cards come out original by construction, not remixed from cards
that already exist.

{{< demo src="/demos/tcg/card_generation_visualiser.html" title="Card generation visualiser: one card's path through the grammar" >}}

## An agent that reads cards

The agent sees the game as a graph, with each card's ability tree hanging off the card, and reads it with a
graph neural network ([GATv2](https://arxiv.org/abs/2105.14491) layers). A card is described only by what it
does, never by an ID, so the agent can play a card the first time it sees it. Trained by self-play, it beats
a simple scripted player in 69% of games on 200 cards it never trained on.

### 1. The board as a graph

{{< demo src="/demos/tcg/gnn_visualiser_1.html" title="GNN visualiser, part 1: the board as a graph" >}}

### 2. One layer of message passing

{{< demo src="/demos/tcg/gnn_visualiser_2.html" title="GNN visualiser, part 2: one GNN layer, step by step" >}}

### 3. Choosing a move

{{< demo src="/demos/tcg/gnn_visualiser_3.html" title="GNN visualiser, part 3: the policy and value heads" >}}

## The first 200 cards

Generated, then costed by a basic balance pass with the agent. Switch to *Thrown away* to see the draws that
failed a test, and why.

{{< demo src="/demos/tcg/card_lab.html" title="Card lab: the first 200 cards" height="85vh" >}}

## What's next

Most abilities are cards a designer might write. A few are much too cheap: *Mire Gossip* draws 4 cards for 1
mana. That's because each effect is priced by its worth in an average deck, and nobody plays an average deck.

Part 2 will tackle three things:

- **A more expressive grammar.** More building blocks for abilities: buffs that last, effects that count
  ("for each Miner you control…"), new triggers and new kinds of card.
- **Controlling synergy.** The generator already makes cards that work together, through shared jobs and
  triggers that feed each other. Part 2 will control it in a more principled way, so each deck has the right
  amount of synergy: enough to reward building around a theme, not so much that one combo wins on its own.
- **Deck generation.** Players build the best deck they can, so part 2 will search for the strongest decks
  and price cards against those, not the average one.

The [repository](https://github.com/mattdown/TCG-Project) has the rest: the rules, training, balancing, related
work, and how to run the demos and the game yourself.

*AI was used to help build this project, write this post and generate the game's assets.*
