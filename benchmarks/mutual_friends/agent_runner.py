#!/usr/bin/env python3
"""
Agent Runner for Mutual Friends Game

Connects LLM agents to play the game automatically and logs the conversation.
"""

import json
import argparse
import time
from typing import List, Dict, Any
from env import MutualFriendsEnv, GameConfig

# Simple LLM client interface
class LLMClient:
    def chat(self, prompt: str) -> str:
        """Override this method for different LLM providers"""
        raise NotImplementedError

class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4", api_key: str = None):
        try:
            import openai
            # If no api_key provided, OpenAI client will automatically use OPENAI_API_KEY env var
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
            else:
                self.client = openai.OpenAI()  # Uses environment variable
            self.model = model
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    def chat(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()

class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: str = None):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
    
    def chat(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

class MockClient(LLMClient):
    """Mock client for testing without API calls"""
    def __init__(self):
        self.turn_count = 0
        self.questions = [
            "Does your person work in a creative field?",
            "Is your person located in Europe?", 
            "Does your person have a bachelor's degree?",
            "Is your person adventurous?"
        ]
    
    def chat(self, prompt: str) -> str:
        # Simple mock behavior
        if self.turn_count < len(self.questions):
            response = self.questions[self.turn_count]
        elif "Danielle" in prompt:  # If we can see the answer in prompt
            response = "[Danielle]"
        else:
            response = "[John]"  # Wrong guess
        
        self.turn_count += 1
        return response

class GameLogger:
    def __init__(self):
        self.logs = []
    
    def log_turn(self, turn: int, player_id: int, action: str, game_state: str = ""):
        entry = {
            "turn": turn,
            "player": f"Player {player_id + 1}",
            "action": action,
            "timestamp": time.time(),
            "game_state": game_state
        }
        self.logs.append(entry)
        print(f"Turn {turn} - Player {player_id + 1}: {action}")
    
    def log_game_end(self, result: str, winner: str = None):
        entry = {
            "event": "game_end",
            "result": result,
            "winner": winner,
            "timestamp": time.time()
        }
        self.logs.append(entry)
        print(f"\nGAME END: {result}")
        if winner:
            print(f"Winner: {winner}")
    
    def save_logs(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"Logs saved to {filename}")

def create_conversation_context(env: MutualFriendsEnv, player_id: int, conversation_history: List[str]) -> str:
    """Create full context for the agent including history"""
    base_prompt = env.get_current_player_prompt()
    
    if conversation_history:
        history_str = "\n".join([f"Turn {i+1}: {msg}" for i, msg in enumerate(conversation_history)])
        context = f"{base_prompt}\n\nConversation so far:\n{history_str}\n\nYour turn:"
    else:
        context = f"{base_prompt}\n\nYou go first. Start by asking a question about traits. Remember: DO NOT GUESS until you are absolutely certain!"
    
    return context

def run_agent_game(env: MutualFriendsEnv, agents: List[LLMClient], logger: GameLogger, max_retries: int = 3):
    """Run a game with LLM agents"""
    conversation_history = []
    
    print("=" * 60)
    print("AGENTS PLAYING MUTUAL FRIENDS")
    print("=" * 60)
    print(f"Mutual friend (hidden): {env.mutual_friend['name']}")
    print(f"Max turns: {env.state.max_turns}")
    print()
    
    while not env.state.done:
        current_player = env.state.current_player_id
        
        # Create context for current agent
        context = create_conversation_context(env, current_player, conversation_history)
        
        # Optional: Print prompts for debugging
        # if env.state.turn == 0:
        #     print(f"\n=== DEBUG: Full Prompt for Player {current_player + 1} ===")
        #     print(context)
        #     print("=== END DEBUG ===\n")
        
        # Get agent response with retries
        action = None
        for attempt in range(max_retries):
            try:
                action = agents[current_player].chat(context)
                break
            except Exception as e:
                print(f"Error getting response from Player {current_player + 1} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    action = "I need to pass this turn"
        
        if not action:
            logger.log_game_end("Game aborted due to agent errors")
            break
        
        # Log the action
        logger.log_turn(env.state.turn + 1, current_player, action)
        conversation_history.append(f"Player {current_player + 1}: {action}")
        
        # Process action in environment
        done, info = env.step(action)
        
        if done:
            outcome = "Both players win" if info.reward > 0 else "Both players lose"
            logger.log_game_end(info.reason, outcome)
            break
    
    return env.get_game_summary()

def main():
    parser = argparse.ArgumentParser(description="Run agents playing Mutual Friends")
    
    # Game config
    parser.add_argument("--game-file", type=str, help="Load game from JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--list-size", type=int, default=5, help="People per list")
    parser.add_argument("--easy", action="store_true", help="Easy mode")
    
    # Agent config
    parser.add_argument("--agent-type", choices=["openai", "anthropic", "mock"], 
                       default="mock", help="LLM provider")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--api-key", type=str, help="API key (or set env var)")
    
    # Personas
    parser.add_argument("--persona-a", type=str, 
                       default="Tech entrepreneur focused on startups",
                       help="Persona for Player A")
    parser.add_argument("--persona-b", type=str,
                       default="Research scientist who values precision", 
                       help="Persona for Player B")
    
    # Output
    parser.add_argument("--log-file", type=str, help="Save logs to file")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")
    
    args = parser.parse_args()
    
    # Create environment
    if args.game_file:
        with open(args.game_file, 'r') as f:
            data = json.load(f)
        config = GameConfig(**data["config"])
        
        # Override personas from command line if provided
        if args.persona_a:
            config.player_a_persona = args.persona_a
        if args.persona_b:
            config.player_b_persona = args.persona_b
            
        env = MutualFriendsEnv(config)
        env.mutual_friend = data["mutual_friend"]
        env.entities_a = data["entities_a"]
        env.entities_b = data["entities_b"]
        env.reset()
    else:
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
    
    # Create agents
    if args.agent_type == "openai":
        agents = [
            OpenAIClient(model=args.model or "gpt-4", api_key=args.api_key),
            OpenAIClient(model=args.model or "gpt-4", api_key=args.api_key)
        ]
    elif args.agent_type == "anthropic":
        agents = [
            AnthropicClient(model=args.model or "claude-3-sonnet-20240229", api_key=args.api_key),
            AnthropicClient(model=args.model or "claude-3-sonnet-20240229", api_key=args.api_key)
        ]
    else:  # mock
        agents = [MockClient(), MockClient()]
    
    # Create logger
    logger = GameLogger()
    
    # Run game
    try:
        summary = run_agent_game(env, agents, logger)
        
        if args.verbose:
            print("\n" + "=" * 60)
            print("GAME SUMMARY")
            print("=" * 60)
            print(json.dumps(summary, indent=2))
        
        if args.log_file:
            logger.save_logs(args.log_file)
    
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
        if args.log_file:
            logger.save_logs(args.log_file)

if __name__ == "__main__":
    main()