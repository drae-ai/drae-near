#!/usr/bin/env python3
"""
Mutual Friends Game Generator

Generates configurable mutual friends games with persona support.
"""

import json
import argparse
from typing import Dict, Any
from env import MutualFriendsEnv, GameConfig

def create_config_from_args(args) -> GameConfig:
    """Create GameConfig from command line arguments"""
    return GameConfig(
        list_size=args.list_size,
        traits_per_person=args.traits_per_person,
        trait_categories=args.trait_categories,
        ambiguity_level=args.ambiguity_level,
        distractor_strength=args.distractor_strength,
        num_strong_distractors=args.num_strong_distractors,
        turn_limit_multiplier=args.turn_limit_multiplier,
        use_llm_judge=not args.no_llm_judge,
        player_a_persona=args.player_a_persona,
        player_b_persona=args.player_b_persona
    )

def generate_game_json(config: GameConfig, seed: int = None) -> Dict[str, Any]:
    """Generate a game and return as JSON-serializable dict"""
    env = MutualFriendsEnv(config)
    env.reset(seed=seed)
    
    return {
        "config": config.__dict__,
        "mutual_friend": env.mutual_friend,
        "entities_a": env.entities_a,
        "entities_b": env.entities_b,
        "player_prompts": {
            "player_a": env._generate_player_prompt(0),
            "player_b": env._generate_player_prompt(1)
        },
        "game_rules": {
            "max_turns": env.state.max_turns,
            "win_condition": "Correctly identify the mutual friend by name",
            "action_format": "Questions: natural language, Guesses: [Name]"
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Generate Mutual Friends game data")
    
    # Game parameters
    parser.add_argument("--list-size", type=int, default=10, help="Number of people per list")
    parser.add_argument("--traits-per-person", type=int, default=4, help="Number of traits per person")
    parser.add_argument("--trait-categories", nargs="+", 
                       default=["profession", "age", "location", "hobby", "education", "personality"],
                       help="Trait categories to use")
    
    # Difficulty parameters
    parser.add_argument("--ambiguity-level", type=float, default=0.7, 
                       help="Trait ambiguity (0=specific, 1=generic)")
    parser.add_argument("--distractor-strength", type=float, default=0.8,
                       help="How similar distractors are to mutual friend")
    parser.add_argument("--num-strong-distractors", type=int, default=2,
                       help="Number of strong distractors per list")
    
    # Turn limit
    parser.add_argument("--turn-limit-multiplier", type=int, default=10,
                       help="Turn limit = multiplier * list_size")
    
    # LLM Judge
    parser.add_argument("--no-llm-judge", action="store_true",
                       help="Disable LLM judge for question validation (faster, less strict)")
    
    # Personas
    parser.add_argument("--player-a-persona", type=str, default=None,
                       help="Persona description for Player A")
    parser.add_argument("--player-b-persona", type=str, default=None,
                       help="Persona description for Player B")
    
    # Output options
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output file (default: stdout)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    
    args = parser.parse_args()
    
    # Generate game
    config = create_config_from_args(args)
    game_data = generate_game_json(config, seed=args.seed)
    
    # Output
    json_str = json.dumps(game_data, indent=2 if args.pretty else None)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(json_str)
        print(f"Game data written to {args.output}")
    else:
        print(json_str)

if __name__ == "__main__":
    main()