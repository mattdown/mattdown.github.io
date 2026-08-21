---
title: "Using AI to understand game dynamics in 1010!"
date: 2026-08-21
mathjax: true
tags: ["Machine Learning", "ML", "AI", "Game Design"]
draft: true
thumbnail: ""
---

# Intro

Today I'm going to take a look at exploring the game dynamics of **1010!** using an AI agent and some interesting 
design metrics. **1010!** is a super successful mobile game that has incredibly simple rules yet a 
huge amount of depth. If you haven't played it, it's probably best described as zero-g tetris. You can place blocks 
wherever you want on the grid. Both vertical and horizontal lines get cleared. The goal, like tetris is to survive 
as long as possible.

But before we dive into 1010!, let's present a bit of wider context as to why we might want to take this approach in
the first place. A long-standing interest of mine is how game agents can be used to explore game design spaces and
provide insight into why some games are super sticky and others are not. Whilst today I'll be looking at a single game, as someone
who works extensively in game prototyping, if we can get an agent to give early insight into possible issues in
long-term game dynamics without the need for extensive human playtesting that's a huge win.

# The Motivation

In order to achieve the goal above we require two main things:

1. A game-playing agent that can play our game to a level representative of human play, it doesn't need to be superhuman and should actually avoid playing in a way that would be counterintuitive to human players.
2. Define some metrics as to what makes a game **fun**. These need to be well-defined things that have a clear 
   numerical definition. 

The agent can then be run tens of thousands of times and can tell us if these metrics are being hit. The designer 
can then make changes before testing again in this very tight feedback loop. In some cases it may even be possible 
for the changes to be made directly by a second agent allowing for some incredibly fast design iteration.

**SHOW DIAGRAM HERE OF FEEDBACK FLOW**

## Defining fun

Let's start with point two. This has been intentionally described in super vague terms. Depending on the type of game 
you're making, this could be a number of different things. But let's keep our discussion today focused on simple 
puzzle games. Here are some of my own thoughts of well-defined metrics that are key for successful puzzle games.

**Complexity of decision-making**

During a typical game are there an overwhelming number of plausible moves that can't easily be narrowed down. An 
example of this being violated might be 19x19 Go to a new player. A game such a Tetris might have a large number of 
possible placements, but it's often fairly easy to narrow to the best few.

**Inevitability of outcome**

Once you're, say, 50% through a typical game. How inevitable is the outcome. If you play as well as you've been 
playing in the first half of the game is the final result effectively set in stone. i.e. in a 2-player game 90% of the 
time the player that's winning at the half-way point goes on to win the game.

Caveat: these metrics are part of the creative process, different designers will naturally want to focus on different 
metrics.

## Game playing agent

In general we need to create an agent that can play not only the base version of the game you're making but also 
variants of the game. Depending on what you're trying to iterate over this could be as simple as playing well after 
relatively minor balance changes to creating an agent that can cope with fairly large rules changes. Of course, the 
gold standard would be an AGI that will could play well on a game it had only just read the rules for. But we don't 
have those yet do we :)

# Exploring 1010! with the above approach

Let's now take a look at 1010! Whilst the approach above was outlined to work with newly developed games, as a 
proof-of-principle I'll look at a game that's already successful 1010!, and see if the design metrics I defined 
above hold.

## Technical setup

Here's the setup:

### Game agent

CNN setup etc. **show technical diagram here**!

### Unity - Python pipeline

**show technical diagram here**!

Unity for the rendering, python for the core logic and training.

Show the local server connection between them

### Calculating and defining metrics

Show how the different metrics were explicitly calculated. 

## Results

Here are the results. But first, here's the unity game with the agent offering hints to give you an idea of how it 
plays.

**embed game here**

### Agent training

Show training results and benchmarks vs. Heurisitc and Random and maybe human?

### Metrics

Show the metric results after 100's of runs

### Bonus

Whilst the main idea of this was to show how 1010! has already been designed to fit these metrics, let's see if we 
can tune the game setup to maybe improve some of these metrics and see if we prefer the newer design or not?

**link to embedded version of game again, with flag to load the new balance etc.**