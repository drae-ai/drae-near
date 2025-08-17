import re, os, random, json
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass
from faker import Faker

# Simple TextArena-like base classes for this implementation
class ObservationType:
    PLAYER_ACTION = "player_action"
    GAME_MESSAGE = "game_message"
    GAME_ACTION_DESCRIPTION = "game_action_description"

class Info:
    def __init__(self, done: bool = False, reward: float = 0, reason: str = ""):
        self.done = done
        self.reward = reward
        self.reason = reason

class GameState:
    def __init__(self, num_players: int, max_turns: int, seed: Optional[int] = None):
        self.num_players = num_players
        self.max_turns = max_turns
        self.turn = 0
        self.current_player_id = 0
        self.done = False
        self.reward = 0
        self.reason = ""
        self.game_state = {}
        self.observations = []
        if seed:
            random.seed(seed)
    
    def add_observation(self, message: str, from_id: int = -1, to_id: int = -1, observation_type: str = ObservationType.GAME_MESSAGE):
        self.observations.append({
            "message": message,
            "from_id": from_id,
            "to_id": to_id,
            "type": observation_type,
            "turn": self.turn
        })
    
    def set_outcome(self, reward: float, reason: str):
        self.done = True
        self.reward = reward
        self.reason = reason
    
    def check_turn_limit(self) -> bool:
        return self.turn >= self.max_turns
    
    def step(self) -> Tuple[bool, Info]:
        self.turn += 1
        self.current_player_id = (self.current_player_id + 1) % self.num_players
        return self.done, Info(self.done, self.reward, self.reason)

@dataclass
class GameConfig:
    list_size: int = 10
    traits_per_person: int = 4
    trait_categories: List[str] = None
    ambiguity_level: float = 0.7
    distractor_strength: float = 0.8
    num_strong_distractors: int = 2
    turn_limit_multiplier: int = 10  # multiplier * list_size = max_turns
    player_a_persona: Optional[str] = None
    player_b_persona: Optional[str] = None
    use_llm_judge: bool = True  # Use LLM to validate questions
    
    def __post_init__(self):
        if self.trait_categories is None:
            self.trait_categories = ["profession", "age", "location", "hobby", "education", "personality"]

class MutualFriendsEnv:
    def __init__(self, config: GameConfig = None):
        self.config = config or GameConfig()
        self.fake = Faker()
        self.entities_a = []
        self.entities_b = []
        self.mutual_friend = None
        self.state = None
        
        # Trait pools for generation
        self.trait_pools = {
            "profession": ["software engineer", "teacher", "doctor", "artist", "chef", "lawyer", "nurse", "mechanic", "writer", "architect"],
            "age": ["20s", "30s", "40s", "50s", "60s"],
            "location": ["San Francisco", "New York", "London", "Tokyo", "Berlin", "Sydney", "Toronto", "Paris", "Amsterdam", "Barcelona"],
            "hobby": ["reading", "hiking", "cooking", "gaming", "music", "sports", "photography", "gardening", "traveling", "painting"],
            "education": ["high school", "bachelor's degree", "master's degree", "PhD", "trade school"],
            "personality": ["outgoing", "introverted", "creative", "analytical", "adventurous", "calm", "energetic", "thoughtful"]
        }
    
    def _generate_traits(self, num_traits: int) -> Dict[str, str]:
        """Generate random traits for a person"""
        selected_categories = random.sample(self.config.trait_categories, min(num_traits, len(self.config.trait_categories)))
        traits = {}
        
        for category in selected_categories:
            if self.config.ambiguity_level > 0.5:
                # High ambiguity - use more generic terms
                traits[category] = random.choice(self.trait_pools[category])
            else:
                # Low ambiguity - use more specific terms
                trait = random.choice(self.trait_pools[category])
                if category == "profession":
                    trait += f" at a {random.choice(['startup', 'corporation', 'nonprofit'])}"
                elif category == "location":
                    trait += f" in the {random.choice(['downtown', 'suburbs', 'outskirts'])} area"
                traits[category] = trait
        
        return traits
    
    def _create_person(self, name: str = None) -> Dict[str, Any]:
        """Create a person with name and traits"""
        if name is None:
            name = self.fake.first_name()
        
        traits = self._generate_traits(self.config.traits_per_person)
        
        return {
            "name": name,
            "traits": traits
        }
    
    def _create_distractor(self, mutual_friend: Dict[str, Any], strength: float) -> Dict[str, Any]:
        """Create a distractor person with specified similarity to mutual friend"""
        name = self.fake.first_name()
        
        # Determine how many traits to share based on strength
        shared_traits_count = int(strength * len(mutual_friend["traits"]))
        
        # Copy some traits from mutual friend
        mutual_traits = list(mutual_friend["traits"].items())
        shared_traits = dict(random.sample(mutual_traits, shared_traits_count))
        
        # Fill remaining traits randomly
        remaining_categories = [cat for cat in self.config.trait_categories 
                             if cat not in shared_traits and len(shared_traits) < self.config.traits_per_person]
        
        for category in remaining_categories[:self.config.traits_per_person - len(shared_traits)]:
            shared_traits[category] = random.choice(self.trait_pools[category])
        
        return {
            "name": name,
            "traits": shared_traits
        }
    
    def generate_game_data(self):
        """Generate the entity lists with one mutual friend"""
        # Create the mutual friend
        self.mutual_friend = self._create_person()
        
        # Generate lists for both players
        self.entities_a = [self.mutual_friend]
        self.entities_b = [dict(self.mutual_friend)]  # Copy for player B
        
        # Add strong distractors
        for _ in range(self.config.num_strong_distractors):
            distractor_a = self._create_distractor(self.mutual_friend, self.config.distractor_strength)
            distractor_b = self._create_distractor(self.mutual_friend, self.config.distractor_strength)
            self.entities_a.append(distractor_a)
            self.entities_b.append(distractor_b)
        
        # Fill remaining slots with random people
        while len(self.entities_a) < self.config.list_size:
            self.entities_a.append(self._create_person())
        
        while len(self.entities_b) < self.config.list_size:
            self.entities_b.append(self._create_person())
        
        # Shuffle the lists
        random.shuffle(self.entities_a)
        random.shuffle(self.entities_b)
    
    def _entities_to_string(self, entities: List[Dict[str, Any]]) -> str:
        """Format entity list for display"""
        formatted = []
        for i, person in enumerate(entities, start=1):
            traits_str = ", ".join([f"{k}: {v}" for k, v in person["traits"].items()])
            formatted.append(f"{i}. {person['name']} ({traits_str})")
        return "\n".join(formatted)
    
    def _generate_player_prompt(self, player_id: int) -> str:
        """Generate prompt for a specific player"""
        entities = self.entities_a if player_id == 0 else self.entities_b
        other_player = "Player 2" if player_id == 0 else "Player 1"
        
        base_prompt = (
            f"You are Player {player_id + 1} in the Mutual Friends game.\n"
            f"You and {other_player} each have a list of people. Exactly ONE person appears on both lists - this is your mutual friend.\n"
            f"Your goal is to identify the mutual friend through conversation.\n\n"
            f"Game Rules:\n"
            f"- You can ASK QUESTIONS about traits using ONLY descriptive language\n"
            f"- You can MAKE GUESSES by wrapping exactly ONE name in brackets (e.g., '[John Smith]')\n"
            f"- NO multiple guesses: '[Jimmy or Kyle]', '[maybe Sarah]', '[either Tom or Mike]' are FORBIDDEN\n"
            f"- Take turns with the other player\n\n"
            f"STRICT RULES - FORBIDDEN:\n"
            f"1. Specific trait values: 'London', 'engineer', 'PhD', '30s'\n"
            f"2. Multiple questions: 'Is our friend a mechanic or lawyer?'\n"
            f"3. Disguised multiple-choice: 'Named after a rockstar or movie character?'\n"
            f"4. Spelling hints: 'starts with B', 'rhymes with'\n"
            f"5. Examples/elaborations: 'Does our friend garden? Like planting roses?'\n\n"
            f"ALLOWED - Single descriptive questions:\n"
            f"- 'Does our friend work in a field that builds things?'\n"
            f"- 'Do they live in a European capital?'\n"
            f"- 'Are they in the earlier part of their career?'\n"
            f"- 'Do they have advanced education?'\n\n"
            f"CRITICAL: Only make a guess when you are ABSOLUTELY CERTAIN. Wrong guesses end the game with both players losing!\n"
            f"Strategy: Use creative descriptions to narrow down possibilities without saying the exact traits.\n\n"
            f"Your list of people:\n{self._entities_to_string(entities)}\n\n"
        )
        
        # Add persona-specific context (make it stronger and more prominent)
        if player_id == 0 and self.config.player_a_persona:
            persona_prompt = f"""
=== YOUR CHARACTER ROLE ===
{self.config.player_a_persona}

CRITICAL: You MUST stay in character at ALL TIMES. Every single question and response must reflect this persona's:
- Vocabulary and slang
- Way of thinking and priorities  
- Cultural references and analogies
- Speech patterns and tone

If you break character even once, you fail the game. Your persona is MORE IMPORTANT than winning.
=== END CHARACTER ROLE ===

"""
            base_prompt = persona_prompt + base_prompt
        elif player_id == 1 and self.config.player_b_persona:
            persona_prompt = f"""
=== YOUR CHARACTER ROLE ===
{self.config.player_b_persona}

CRITICAL: You MUST stay in character at ALL TIMES. Every single question and response must reflect this persona's:
- Vocabulary and slang
- Way of thinking and priorities
- Cultural references and analogies  
- Speech patterns and tone

If you break character even once, you fail the game. Your persona is MORE IMPORTANT than winning.
=== END CHARACTER ROLE ===

"""
            base_prompt = persona_prompt + base_prompt
        
        return base_prompt
    
    def reset(self, seed: Optional[int] = None):
        """Initialize a new game"""
        if seed:
            random.seed(seed)
            self.fake.seed_instance(seed)
        
        # Generate game data
        self.generate_game_data()
        
        # Determine turn limit based on list size
        max_turns = self.config.turn_limit_multiplier * self.config.list_size
        
        # Initialize game state
        self.state = GameState(num_players=2, max_turns=max_turns, seed=seed)
        self.state.game_state = {
            "mutual_friend": self.mutual_friend,
            "entities_a": self.entities_a,
            "entities_b": self.entities_b,
            "player_prompts": {
                0: self._generate_player_prompt(0),
                1: self._generate_player_prompt(1)
            }
        }
    
    def _is_forbidden_question(self, action: str) -> bool:
        """Use LLM judge to check if question violates game rules"""
        judge_prompt = f"""
You are a judge for the Mutual Friends game. Check if this violates ANY of these rules:

RULE 1 - NO specific trait values:
FORBIDDEN: "Does our friend live in London?", "Is our friend an engineer?", "Does our friend have a PhD?"
ALLOWED: "Does our friend live in a European capital?", "Does our friend work in a field that builds things?"

RULE 2 - NO multiple questions in one turn:
FORBIDDEN: "Does our friend work with cars, or is he more into legal documents?"
FORBIDDEN: "Is our friend a mechanic or a lawyer?"
ALLOWED: "Does our friend work in a hands-on profession?"

RULE 3 - NO disguised multiple-choice questions:
FORBIDDEN: "Is our friend named after a rockstar or a movie character?"
FORBIDDEN: "Does our friend have a name like a famous person or a common name?"
ALLOWED: "Does our friend have an unusual name?"

RULE 4 - NO spelling hints:
FORBIDDEN: "Does our friend's name start with J?", "Does it rhyme with..."

RULE 5 - NO specific examples or elaborations:
FORBIDDEN: "Does our friend enjoy gardening? You know, like planting roses and stuff?"
FORBIDDEN: "Does our friend work with tools? Like hammers and wrenches?"
FORBIDDEN: "Does our friend live somewhere cold? Like with snow and ice?"
ALLOWED: "Does our friend enjoy gardening?"
ALLOWED: "Does our friend work with tools?"
ALLOWED: "Does our friend live somewhere cold?"

Question to evaluate: "{action}"

Does this violate ANY rule? Answer with just: FORBIDDEN or ALLOWED
"""
        
        try:
            # Use a simple LLM client for judging
            # You could use OpenAI, Anthropic, or any other LLM here
            import openai
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Cheaper model for judging
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                max_tokens=10
            )
            judgment = response.choices[0].message.content.strip().upper()
            return judgment == "FORBIDDEN"
        except Exception as e:
            # Fallback: if LLM judge fails, be permissive
            print(f"Warning: LLM judge failed ({e}), allowing question")
            return False
    
    def _is_multiple_name_guess(self, guess: str) -> bool:
        """Check if guess contains multiple names or uncertainty"""
        guess_lower = guess.lower()
        
        # Check for multiple name patterns
        multiple_patterns = [
            r'\bor\b',           # "Jimmy or Kyle"
            r'\beither\b',       # "either Jimmy or Kyle"  
            r'\bmaybe\b',        # "maybe Jimmy"
            r'\bperhaps\b',      # "perhaps Kyle"
            r'\bpossibly\b',     # "possibly Sarah"
            r'\bcould be\b',     # "could be Jimmy"
            r'\bmight be\b',     # "might be Kyle"
            r'\band\b',          # "Jimmy and Kyle" 
            r',',                # "Jimmy, Kyle"
            r'/',                # "Jimmy/Kyle"
        ]
        
        for pattern in multiple_patterns:
            if re.search(pattern, guess_lower):
                return True
        
        # Check if there are multiple capitalized words (likely names)
        # But allow common name patterns like "John Smith", "Mary Jane"
        words = guess.split()
        capitalized_words = [w for w in words if w and w[0].isupper()]
        
        # If more than 2 capitalized words, likely multiple names
        if len(capitalized_words) > 2:
            return True
            
        return False
    
    def step(self, action: str) -> Tuple[bool, Info]:
        """Process a player's action"""
        player_id = self.state.current_player_id
        self.state.add_observation(message=action, from_id=player_id, observation_type=ObservationType.PLAYER_ACTION)
        
        # Check if action is a guess (wrapped in brackets)
        guess_match = re.search(r'\[([^\]]+)\]', action)
        
        if guess_match:
            guessed_name = guess_match.group(1).strip()
            
            # Check if guess contains multiple names (forbidden)
            if self._is_multiple_name_guess(guessed_name):
                self.state.add_observation(
                    message=f"INVALID GUESS: Player {player_id + 1}, you must guess exactly ONE person's name. Multiple names, 'or', 'either', etc. are not allowed. Please make a single, specific guess.",
                    observation_type=ObservationType.GAME_MESSAGE
                )
            else:
                # Valid single name guess - check if correct
                if guessed_name.lower() == self.mutual_friend["name"].lower():
                    self.state.set_outcome(
                        reward=1.0, 
                        reason=f"SUCCESS! Both players win - {self.mutual_friend['name']} was correctly identified as the mutual friend!"
                    )
                else:
                    self.state.set_outcome(
                        reward=0.0,
                        reason=f"FAILURE: Incorrect guess '{guessed_name}'. The mutual friend was {self.mutual_friend['name']}. Both players lose."
                    )
        else:
            # It's a question - check if it's forbidden (if judge is enabled)
            if self.config.use_llm_judge and self._is_forbidden_question(action):
                self.state.add_observation(
                    message=f"JUDGE RULING: Player {player_id + 1}, your question uses forbidden specific trait mentions. Please rephrase using only descriptive language. Example: Instead of 'Does our friend live in London?', ask 'Does our friend live in a major European city?'",
                    observation_type=ObservationType.GAME_MESSAGE
                )
            else:
                # Valid question - acknowledge it
                self.state.add_observation(
                    message=f"Player {player_id + 1} asked: {action}",
                    observation_type=ObservationType.GAME_ACTION_DESCRIPTION
                )
        
        # Check turn limit
        if self.state.check_turn_limit() and not self.state.done:
            self.state.set_outcome(
                reward=0.0,
                reason=f"Game ended after {self.state.max_turns} turns. The mutual friend was {self.mutual_friend['name']}."
            )
        
        return self.state.step()
    
    def get_current_player_prompt(self) -> str:
        """Get the prompt for the current player"""
        return self.state.game_state["player_prompts"][self.state.current_player_id]
    
    def get_game_summary(self) -> Dict[str, Any]:
        """Get a summary of the game state"""
        return {
            "config": self.config.__dict__,
            "mutual_friend": self.mutual_friend,
            "current_turn": self.state.turn if self.state else 0,
            "max_turns": self.state.max_turns if self.state else 0,
            "done": self.state.done if self.state else False,
            "success": self.state.reward > 0 if self.state and self.state.done else None,
            "outcome": "Both players win" if self.state and self.state.done and self.state.reward > 0 else ("Both players lose" if self.state and self.state.done else "Game in progress")
        }