"""DevelopMap: Universal Product Taxonomy for Developmental Domains.

This module defines the mapping between retail products and developmental
domains based on established screening tools (M-CHAT, ASQ, SPM).
"""

from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class DevelopmentalDomain:
    """Represents a developmental domain with associated keywords and patterns."""
    name: str
    description: str
    keywords: Set[str]
    example_products: List[str]
    clinical_alignment: str


class DevelopMap:
    """Universal product taxonomy for developmental screening."""
    
    def __init__(self):
        self.domains = self._initialize_domains()
        
    def _initialize_domains(self) -> Dict[str, DevelopmentalDomain]:
        """Initialize all developmental domains with keywords and patterns."""
        
        domains = {
            "fine_motor": DevelopmentalDomain(
                name="Fine Motor Development",
                description="Hand-eye coordination, dexterity, manipulation skills",
                keywords={
                    "puzzle", "puzzles", "block", "blocks", "bead", "beads",
                    "threading", "lacing", "crayon", "crayons", "marker", "markers",
                    "playdough", "play-doh", "clay", "scissors", "cutting",
                    "stacking", "nesting", "peg", "pegs", "tweezers", "tongs",
                    "fine motor", "dexterity", "manipulation", "grasp", "pincer"
                },
                example_products=[
                    "Wooden puzzles", "Building blocks", "Bead maze",
                    "Threading toys", "Crayons", "Playdough set"
                ],
                clinical_alignment="ASQ Fine Motor Scale"
            ),
            
            "gross_motor": DevelopmentalDomain(
                name="Gross Motor Development",
                description="Large muscle movement, balance, coordination",
                keywords={
                    "bike", "bicycle", "tricycle", "scooter", "ball", "balls",
                    "trampoline", "jumping", "climbing", "slide", "swing",
                    "ride-on", "push toy", "walker", "balance", "coordination",
                    "gross motor", "active play", "outdoor", "sports", "running"
                },
                example_products=[
                    "Balance bike", "Scooter", "Soccer ball", "Trampoline",
                    "Climbing dome", "Ride-on toy"
                ],
                clinical_alignment="ASQ Gross Motor Scale"
            ),
            
            "language": DevelopmentalDomain(
                name="Language Development",
                description="Communication, vocabulary, speech, literacy",
                keywords={
                    "book", "books", "flashcard", "flashcards", "speech",
                    "language", "vocabulary", "communication", "aac", "pecs",
                    "picture cards", "word", "words", "letter", "letters",
                    "alphabet", "phonics", "reading", "story", "stories",
                    "talking", "verbal", "articulation", "pronunciation"
                },
                example_products=[
                    "Board books", "Flashcards", "AAC device", "PECS cards",
                    "Speech therapy tools", "Alphabet toys"
                ],
                clinical_alignment="ASQ Communication Scale, M-CHAT Language Items"
            ),
            
            "social_emotional": DevelopmentalDomain(
                name="Social-Emotional Development",
                description="Social interaction, emotional regulation, pretend play",
                keywords={
                    "social", "emotion", "emotions", "feelings", "pretend",
                    "play", "doll", "dolls", "action figure", "figures",
                    "board game", "game", "cooperative", "turn-taking",
                    "friends", "friendship", "empathy", "sharing", "caring",
                    "role play", "dress-up", "costume", "puppet", "puppets"
                },
                example_products=[
                    "Board games", "Dolls", "Pretend play sets", "Social stories",
                    "Emotion cards", "Puppet theater"
                ],
                clinical_alignment="ASQ Personal-Social Scale, M-CHAT Social Items"
            ),
            
            "sensory": DevelopmentalDomain(
                name="Sensory Processing",
                description="Sensory integration, regulation, sensitivities",
                keywords={
                    "sensory", "fidget", "fidgets", "weighted", "blanket",
                    "compression", "chew", "chewable", "chewy", "texture",
                    "tactile", "noise-canceling", "headphones", "earplugs",
                    "calming", "soothing", "regulation", "vestibular",
                    "proprioceptive", "sensory bin", "sensory toys"
                },
                example_products=[
                    "Fidget toys", "Weighted blanket", "Noise-canceling headphones",
                    "Chew necklace", "Sensory bin", "Compression vest"
                ],
                clinical_alignment="Sensory Processing Measure (SPM)"
            ),
            
            "adaptive": DevelopmentalDomain(
                name="Adaptive Equipment",
                description="Special needs equipment, assistive devices",
                keywords={
                    "adaptive", "special needs", "therapy", "therapeutic",
                    "occupational", "physical therapy", "assistive",
                    "special utensils", "non-slip", "positioning", "support",
                    "orthotics", "braces", "walker", "stander", "specialized"
                },
                example_products=[
                    "Special utensils", "Non-slip plates", "Positioning aids",
                    "Therapy ball", "Orthotics", "Adaptive scissors"
                ],
                clinical_alignment="Adaptive Behavior Assessment System (ABAS)"
            ),
            
            "sleep": DevelopmentalDomain(
                name="Sleep Management",
                description="Sleep aids, bedtime routine, sleep regulation",
                keywords={
                    "sleep", "bedtime", "night", "nap", "sleeping",
                    "white noise", "sound machine", "night light", "nightlight",
                    "sleep clock", "ok to wake", "melatonin", "calming",
                    "soothing", "lullaby", "sleep sack", "swaddle", "routine"
                },
                example_products=[
                    "White noise machine", "Night light", "Sleep clock",
                    "Weighted blanket", "Sleep sack", "Melatonin supplements"
                ],
                clinical_alignment="Children's Sleep Habits Questionnaire (CSHQ)"
            ),
            
            "feeding": DevelopmentalDomain(
                name="Feeding Challenges",
                description="Feeding difficulties, oral motor, picky eating",
                keywords={
                    "feeding", "picky eater", "picky eating", "food",
                    "oral motor", "chewing", "swallowing", "texture",
                    "sensory food", "divided plate", "suction", "sippy",
                    "straw", "bottle", "spoon", "fork", "utensils",
                    "supplements", "nutrition", "meal", "eating"
                },
                example_products=[
                    "Divided plates", "Sensory bottles", "Texture foods",
                    "Oral motor tools", "Supplements", "Special utensils"
                ],
                clinical_alignment="Pediatric Feeding Assessment"
            ),
            
            "behavioral": DevelopmentalDomain(
                name="Behavioral Regulation",
                description="Behavior management, routines, executive function",
                keywords={
                    "behavior", "behavioral", "routine", "schedule",
                    "visual schedule", "timer", "countdown", "reward",
                    "chart", "token", "sticker", "organization", "planner",
                    "executive function", "attention", "focus", "calm down",
                    "regulation", "self-control", "impulse"
                },
                example_products=[
                    "Visual schedules", "Timers", "Reward charts", "Token boards",
                    "Countdown clocks", "Organization tools"
                ],
                clinical_alignment="Behavior Assessment System for Children (BASC)"
            ),
            
            "therapeutic": DevelopmentalDomain(
                name="Therapeutic Resources",
                description="Therapy materials, assessment tools, parent resources",
                keywords={
                    "therapy", "therapist", "therapeutic", "intervention",
                    "assessment", "screening", "milestone", "developmental",
                    "workbook", "parent guide", "resource", "training",
                    "early intervention", "special education", "iep", "ifsp"
                },
                example_products=[
                    "Therapy workbooks", "Parent guides", "Assessment tools",
                    "Milestone trackers", "Intervention materials"
                ],
                clinical_alignment="Various screening and intervention tools"
            )
        }
        
        return domains
    
    def get_domain_names(self) -> List[str]:
        """Return list of all domain names."""
        return list(self.domains.keys())
    
    def get_domain(self, domain_name: str) -> DevelopmentalDomain:
        """Get a specific domain by name."""
        return self.domains.get(domain_name)
    
    def get_all_keywords(self) -> Dict[str, Set[str]]:
        """Return all keywords organized by domain."""
        return {name: domain.keywords for name, domain in self.domains.items()}
    
    def get_clinical_alignments(self) -> Dict[str, str]:
        """Return clinical tool alignments for each domain."""
        return {name: domain.clinical_alignment for name, domain in self.domains.items()}


# Global instance
DEVELOPMAP = DevelopMap()