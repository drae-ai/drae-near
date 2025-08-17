#!/usr/bin/env python3
"""
Mutual Friends Game Runner

Simple CLI runner for testing the mutual friends game manually.
"""

import json
import argparse
from env import MutualFriendsEnv, GameConfig

def load_game_from_json(filename: str) -> MutualFriendsEnv:
    """Load a game from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Reconstruct config
    config_data = data["config"]
    config = GameConfig(**config_data)
    
    # Create environment
    env = MutualFriendsEnv(config)
    
    # Set the generated data
    env.mutual_friend = data["mutual_friend"]
    env.entities_a = data["entities_a"]
    env.entities_b = data["entities_b"]
    
    # Initialize state
    env.reset()
    
    return env

def print_game_info(env: MutualFriendsEnv):
    """Print game information"""
    print("=" * 60)
    print("MUTUAL FRIENDS GAME")
    print("=" * 60)
    print(f"Turn limit: {env.state.max_turns}")
    print(f"Mutual friend (hidden): {env.mutual_friend['name']}")
    print()
    
    print("PLAYER LISTS:")
    print("-" * 30)
    print("Player 1 list:")
    print(env._entities_to_string(env.entities_a))
    print()
    print("Player 2 list:")
    print(env._entities_to_string(env.entities_b))
    print()

def interactive_game(env: MutualFriendsEnv):
    """Run an interactive game session"""
    print_game_info(env)
    
    print("GAME RULES:")
    print("- Ask questions about traits: 'Does your person work in tech?'")
    print("- Make guesses with brackets: '[John Smith]'")
    print("- Type 'quit' to exit")
    print("=" * 60)
    
    while not env.state.done:
        current_player = env.state.current_player_id + 1
        print(f"\nTurn {env.state.turn + 1}/{env.state.max_turns}")
        print(f"Player {current_player}'s turn:")
        
        # Show player's prompt
        prompt = env.get_current_player_prompt()
        print(f"\n{prompt}\n")
        
        # Get action
        action = input(f"Player {current_player} action: ").strip()
        
        if action.lower() == 'quit':
            break
        
        if not action:
            print("Please enter an action.")
            continue
        
        # Process action
        done, info = env.step(action)
        
        print(f"Action: {action}")
        
        if done:
            print(f"\nGAME OVER!")
            print(f"Result: {info.reason}")
            print(f"Score: {info.reward}")
            break
    
    # Show game summary
    summary = env.get_game_summary()
    print(f"\nGAME SUMMARY:")
    print(f"Turns taken: {summary['current_turn']}")
    print(f"Winner: {summary.get('winner', 'None')}")
    print(f"Mutual friend was: {summary['mutual_friend']['name']}")

def main():
    parser = argparse.ArgumentParser(description="Run Mutual Friends game")
    parser.add_argument("--game-file", type=str, help="Load game from JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Quick config options
    parser.add_argument("--list-size", type=int, default=5, help="Number of people per list")
    parser.add_argument("--easy", action="store_true", help="Easy mode (fewer distractors)")
    parser.add_argument("--persona-a", type=str, help="Persona for Player A")
    parser.add_argument("--persona-b", type=str, help="Persona for Player B")
    
    args = parser.parse_args()
    
    if args.game_file:
        env = load_game_from_json(args.game_file)
    else:
        # Create quick config
        config = GameConfig(
            list_size=args.list_size,
            traits_per_person=3,
            distractor_strength=0.5 if args.easy else 0.8,
            num_strong_distractors=1 if args.easy else 2,
            player_a_persona=args.persona_a,
            player_b_persona=args.persona_b
        )
        env = MutualFriendsEnv(config)
        env.reset(seed=args.seed)
    
    interactive_game(env)

if __name__ == "__main__":
    main()