"""
ResearchFlow — Input Guardrails Middleware

Detects and blocks prompt injection / stuffing attacks
in user inputs before they reach the agent pipeline.
"""
import re

override_patterns = [
    r"(?i)ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
    r"(?i)disregard\s+(?:all\s+)?(?:previous|prior)\s+instructions",
    r"(?i)system\s+(?:override|reset|reboot)",
    r"(?i)you\s+are\s+now\s+(?:a|an)\s+",
    r"(?i)acting\s+as\s+",
    r"(?i)new\s+rule[s]?\s*[:\-]"
]

stuffing_patterns = [
    r"###\s*Instruction",
    r"\[INST\]", # Marks the beginning of instructions from user
    r"\[\/INST\]", # Marks the end of instructions from user
    r"(?i)end\s+of\s+context",
    r"(?i)now\s+(?:do|answer)\s+the\s+following",
    r"={3,}", # Long lines of equals signs often used as fake separators
    r"-{3,}"  # Long lines of dashes
]

injection_markers = [
    r"(?i)###\s*Instruction:?",
    r"(?i)###\s*Response:?",
    r"\[/?INST\]",
    r"(?i)system\s*prompt:?",
    r"(?i)user\s*input:?"
]

def detect_injection(user_input: str) -> bool:
    """
    Scan user input for common prompt injection patterns.

    TODO:
    - Check for system prompt override attempts.
    - Check for instruction stuffing patterns.
    - Return True if injection is detected, False otherwise.
    """
    all_patterns = override_patterns + stuffing_patterns
    
    for pattern in all_patterns:
        if re.search(pattern, user_input):
            return True
            
    return False


def sanitize_input(user_input: str) -> str:
    """
    Clean user input by removing or escaping dangerous patterns.

    TODO:
    - Strip known injection markers.
    - Escape special formatting that could manipulate prompts.
    - Return the sanitized string.
    """
    
    sanitized = user_input
    for pattern in injection_markers:
        sanitized = re.sub(pattern, "", sanitized)
    
    # Neutralize repeated structural delimiters (3+ characters)
    sanitized = re.sub(r"([=\-]{3,})", r" \1 ", sanitized)

    # Prevent "Instruction Stuffing" via excessive newlines
    # Attacks often use many newlines to "push" the real system prompt out of view
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)

    return sanitized.strip()
