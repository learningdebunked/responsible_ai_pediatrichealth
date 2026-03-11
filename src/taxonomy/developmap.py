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
                clinical_alignment="ASQ-3 Fine Motor domain (Squires & Bricker, 2009); Peabody Developmental Motor Scales-2 (Folio & Fewell, 2000)"
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
                clinical_alignment="ASQ-3 Gross Motor domain (Squires & Bricker, 2009); Bayley-III Motor Scale (Bayley, 2006)"
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
                clinical_alignment="ASQ-3 Communication domain (Squires & Bricker, 2009); M-CHAT-R/F Items 5,6 (Robins et al., 2014); LENA developmental snapshot"
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
                clinical_alignment="ASQ-3 Personal-Social domain (Squires & Bricker, 2009); M-CHAT-R/F Items 2,7,9,13-15 (Robins et al., 2014); ITSEA Social-Emotional (Carter & Briggs-Gowan, 2006)"
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
                clinical_alignment="Sensory Processing Measure (SPM; Parham & Ecker, 2007); Sensory Profile-2 (Dunn, 2014); Short Sensory Profile (McIntosh et al., 1999)"
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
                clinical_alignment="Adaptive Behavior Assessment System-3 (ABAS-3; Harrison & Oakland, 2015); Vineland-3 Adaptive Behavior Scales (Sparrow et al., 2016)"
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
                clinical_alignment="Children's Sleep Habits Questionnaire (CSHQ; Owens et al., 2000); Brief Infant Sleep Questionnaire (BISQ; Sadeh, 2004)"
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
                clinical_alignment="Pediatric Eating Assessment Tool (PediEAT; Thoyre et al., 2014); Brief Autism Mealtime Behavior Inventory (BAMBI; Lukens & Linscheid, 2008)"
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
                clinical_alignment="BASC-3 (Reynolds & Kamphaus, 2015); Conners Early Childhood (Conners, 2009); BRIEF-P (Gioia et al., 2003)"
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
                clinical_alignment="IDEA Part C (34 CFR 303); AAP developmental surveillance guidelines (Lipkin et al., 2020); CDC Learn the Signs. Act Early."
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


    def validate_against_asq3(self, asq3_domain_map: Dict[str, List[str]] = None) -> Dict[str, dict]:
        """Validate DevelopMap domains against ASQ-3 screening domains.

        ASQ-3 has 5 domains: Communication, Gross Motor, Fine Motor,
        Problem Solving, and Personal-Social. This method checks that
        each DevelopMap domain maps to at least one ASQ-3 domain and
        reports coverage gaps.

        Args:
            asq3_domain_map: Optional override mapping ASQ-3 domain names
                to lists of DevelopMap domain names. If None, uses the
                default mapping below.

        Returns:
            Dict per DevelopMap domain with keys:
                asq3_domains: list of mapped ASQ-3 domains
                has_asq3_mapping: bool
                clinical_ref: extracted citation string
        """
        if asq3_domain_map is None:
            asq3_domain_map = {
                'Communication': ['language'],
                'Gross Motor': ['gross_motor'],
                'Fine Motor': ['fine_motor'],
                'Problem Solving': ['sensory', 'behavioral', 'adaptive'],
                'Personal-Social': ['social_emotional', 'feeding', 'sleep'],
            }

        # Invert: DevelopMap domain -> ASQ-3 domains
        dm_to_asq = {}
        for asq_domain, dm_domains in asq3_domain_map.items():
            for dm in dm_domains:
                dm_to_asq.setdefault(dm, []).append(asq_domain)

        results = {}
        for domain_name, domain in self.domains.items():
            asq_domains = dm_to_asq.get(domain_name, [])
            results[domain_name] = {
                'asq3_domains': asq_domains,
                'has_asq3_mapping': len(asq_domains) > 0,
                'clinical_ref': domain.clinical_alignment,
            }

        # Summary
        n_mapped = sum(1 for v in results.values() if v['has_asq3_mapping'])
        n_total = len(results)
        print(f"ASQ-3 Validation: {n_mapped}/{n_total} DevelopMap domains mapped")
        unmapped = [k for k, v in results.items() if not v['has_asq3_mapping']]
        if unmapped:
            print(f"  Unmapped domains: {unmapped}")
            print(f"  (These domains extend beyond ASQ-3 scope, e.g., therapeutic)")

        return results


# Global instance
DEVELOPMAP = DevelopMap()