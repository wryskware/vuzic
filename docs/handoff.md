# Latent Music Terrarium

## Project Handoff

### Overview

Latent Music Terrarium is a browser-based audiovisual experience that turns music into an evolving simulated world.

Rather than reacting only to basic audio features such as volume, bass, or tempo, the system uses a music-analysis model to capture higher-level relationships within a track. These may include recurring motifs, changes in mood, structural transitions, similarity between sections, and shifts in musical texture.

The resulting analysis drives a living visual environment made from particles, organisms, fields, landscapes, or other emergent systems.

### Core Idea

Each song produces a unique “dream” shaped by its underlying musical structure.

Similar musical moments should create related visual states. A returning chorus may bring back a recognizable environment, species, or movement pattern, while variations in the music cause that state to evolve.

The goal is not to create a literal visualization of the audio. It is to create the feeling that the music is being interpreted by an artificial ecosystem with memory.

### Experience

A user selects a pre-analyzed song and watches or explores its generated visual world in the browser.

The experience may be:

* watched as an automated music video
* explored interactively
* replayed with different visual seeds or interpretations
* shared as a link
* exported as a finished video

### High-Level Structure

The project has two main parts.

**Music Analysis**

A song is processed ahead of time using an audio or music model. The analysis is converted into a compact timeline describing how the musical state changes throughout the track.

**Visual Simulation**

A WebGPU-based renderer uses that timeline to drive an evolving world in real time. The simulation maintains its own history and internal state, allowing the visuals to develop continuously rather than changing independently from moment to moment.

### Design Goals

* Produce visuals that reflect musical structure, not just immediate sound levels.
* Create recognizable visual continuity when musical ideas repeat.
* Support large, fluid, GPU-driven simulations in the browser.
* Make finished experiences easy to share and replay.
* Allow many different visual systems to interpret the same analyzed track.
* Keep the project focused on experimentation, emergence, and audiovisual atmosphere rather than conventional gameplay.

### Initial Scope

The first version should focus on proving the central concept with:

* one analyzed song
* one visual simulation
* one deterministic visual seed
* browser playback
* synchronization between the music and its latent timeline

The primary question is whether model-derived musical representations can produce visuals that feel more coherent, expressive, and memorable than a traditional audio-reactive visualizer.

### Long-Term Possibilities

The project could later grow into a larger section of Vuzic containing multiple simulations, visual styles, shareable dreams, custom song processing, interactive controls, and tools for generating music videos.

At its heart, the project is an experiment in letting music become a place.

