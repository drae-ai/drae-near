# Mutual Friends Game

A configurable LLM game for testing semantic communication between AI agents.

## Overview

Two agents each receive a list of people with traits. Exactly one person appears on both lists - the "mutual friend". Agents must identify this shared person through conversation.

## Key Features

- **Persona-based difficulty**: Agents with personas need fewer turns (12 vs 25)
- **Configurable traits**: Profession, location, age, hobbies, etc.
- **Strong distractors**: Similar people that make identification harder
- **Realistic names**: Generated using Faker library
- **Full logging**: Track conversation and decision-making

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Generate a game
python generate_game.py --seed 42 --pretty

# Test with mock agents (no API needed)
python agent_runner.py --seed 42 --agent-type mock --verbose

# Run with real LLM agents
python agent_runner.py --agent-type openai --api-key YOUR_KEY --log-file game.json
```

## Game Configuration

### Basic Parameters
- `--list-size`: Number of people per list (default: 10)
- `--traits-per-person`: Trait count per person (default: 4) 
- `--seed`: Random seed for reproducibility

### Difficulty Control
- `--ambiguity-level`: Trait specificity (0=specific, 1=generic, default: 0.7)
- `--distractor-strength`: Similarity to mutual friend (0-1, default: 0.8)
- `--num-strong-distractors`: Count of similar people (default: 2)

### Personas (reduces turn limit)
- `--player-a-persona`: Description for Player A
- `--player-b-persona`: Description for Player B

## Examples

### Generate persona-based game
```bash
python generate_game.py \
  --player-a-persona "Tech entrepreneur focused on Silicon Valley culture" \
  --player-b-persona "Academic researcher who values methodical approaches" \
  --output game_personas.json --pretty
```

### Run agents with custom config
```bash
python agent_runner.py \
  --game-file game_personas.json \
  --agent-type openai \
  --model gpt-4 \
  --log-file results.json \
  --verbose
```

### Easy mode for testing
```bash
python agent_runner.py --easy --seed 123 --agent-type mock
```

## Game Mechanics

### Turn Structure
1. **Questions**: "Does your person work in tech?"
2. **Guesses**: "[John Smith]" (wrapped in brackets)
3. **Win condition**: Correctly identify mutual friend by name
4. **Turn limits**: 25 without personas, 12 with personas

### Example Output
```
Turn 1 - Player 1: Does your person work in a creative field?
Turn 2 - Player 2: Yes, they do. Is your person located in Europe?
Turn 3 - Player 1: Yes, London specifically. Do they have a bachelor's degree?
Turn 4 - Player 2: Yes. Is your person adventurous?
Turn 5 - Player 1: [Danielle]

GAME END: Player 1 correctly identified Danielle as the mutual friend!
```

## Agent Types

### Mock Agent (`--agent-type mock`)
- No API calls required
- Simple rule-based responses
- Good for testing game mechanics

### OpenAI (`--agent-type openai`)
```bash
export OPENAI_API_KEY=your_key
python agent_runner.py --agent-type openai --model gpt-4
```

### Anthropic (`--agent-type anthropic`)
```bash
export ANTHROPIC_API_KEY=your_key  
python agent_runner.py --agent-type anthropic --model claude-3-sonnet-20240229
```

## Files

- `env.py`: Game environment and logic
- `generate_game.py`: Create game configurations
- `agent_runner.py`: Run LLM agents automatically
- `run_game.py`: Interactive human testing
- `requirements.txt`: Dependencies

## Research Applications

This framework enables experiments on:
- **Semantic alignment**: How well agents understand each other
- **Communication efficiency**: Turn count analysis
- **Persona effects**: How background knowledge helps
- **Failure modes**: Where communication breaks down

Perfect for testing DRAE protocol vs baseline language models!