---
title: "Hard Things"
date: 2024-08-23T16:08:03-04:00
draft: true
type: "blog"
---

Basically, why to do hard things and how to do them.

Given that the entire point of standard supervised learning is to learn a task by sampling instances of that task along a distribution, it naively seems reasonable to want to formulate an entirely different learning framework.

However, with large scale autoregressive language modeling, we see that _multiple_ tasks can be learned _implicitly_ just by modeling the distribution of language.

The task with ARC is, fundamentally, to learn a task that is, by construction, out of the distribution of the autoregressive task.

Another perspective one can have on this is that ARC is really testing _meta learning_ capabilities, or also few-shot learning capabilities, given human priors.

To clarify, it is against the spirit of the challenge to be solving a challenge by querying, or retrieving some solution that to a task was learned during training. The goal is to build a very capable sample-efficient learner that can learn these new tasks "on the fly".

I guess there are a many goals of this. One is to build the sample-efficient learner. The other is to build some reasoning agent or system that is capable of "abstraction". Although, abstraction itself can be another anthropomorphic bias. IE perhaps it is possible to efficiently build a sample efficient learner that does not utilize abstraction whatsoever.

Another important human bias that is in the challenge is the "Knowledge Priors". These knowledge priors are again human-centric, and are our interpretation of solutions to these challenges. However, this is one of many solutions in solution space. AI's solution needn't follow this reasoning, or be explainable. But it would be useful if it were.

The question now is: can models be trained autoregressively at scale to: _implicitly_ learn how to learn? Learn how to be taught new skills? New priors? Update knowledge spaces?

This may require a new framework... perhaps we cannot abstract away learners. But that does not make sense as a next step right now, we have to find the nearest step.

I suppose there is a vast literature of RL looking at the same space of challenges. IE how do we create sample efficient learners, are neural networks good sample efficient learners?

Also, multimodal models may have more information to reason about.

Today I also learn that neural networks in a certain perspective of framework can be often equivalently refactored to another perspective, therefore giving multiple viewpoints of the same topics.

Well, humans don't necessarily learn from scratch a new skill every time. They have to draw upon prior knowledge in order to build "new" or "novel" things. I guess this means that the system can as well.
