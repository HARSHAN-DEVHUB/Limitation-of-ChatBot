from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


INTENTS = [
    {
        "name": "portal_password_reset",
        "user_q": "I cannot access my student portal account.",
        "agent_a": "Use Forgot Password and verify through your university email.",
        "user_follow": "What if I cannot access my university email?",
        "agent_follow": "Visit IT helpdesk with student ID and national ID for manual verification.",
    },
    {
        "name": "late_registration",
        "user_q": "Can I do late course registration?",
        "agent_a": "Late registration is allowed within 7 days with department approval.",
        "user_follow": "Where should I submit the approval?",
        "agent_follow": "Upload the approval letter in the portal under Academics > Late Registration Request.",
    },
    {
        "name": "exam_deadlines",
        "user_q": "What is the deadline to apply for final exams?",
        "agent_a": "Exam application closes 14 days before the first exam date.",
        "user_follow": "Can I apply after the deadline?",
        "agent_follow": "Late applications are accepted for 3 days with a late fee and faculty approval.",
    },
    {
        "name": "fee_payment",
        "user_q": "How can I pay semester fees?",
        "agent_a": "Pay fees through the student finance portal using card or bank transfer.",
        "user_follow": "What happens if I miss the payment deadline?",
        "agent_follow": "A late fee applies after deadline and registration may be temporarily blocked.",
    },
    {
        "name": "hostel_application",
        "user_q": "How do I apply for hostel accommodation?",
        "agent_a": "Submit hostel application through campus services before the accommodation deadline.",
        "user_follow": "What documents are required?",
        "agent_follow": "Upload student ID, guardian consent, and medical declaration form.",
    },
]


def main(output: Path, conversations: int, seed: int) -> None:
    random.seed(seed)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for idx in range(conversations):
            intent = random.choice(INTENTS)
            doc_id = f"synthetic_uni_{idx // 250:03d}"
            conv_id = f"syn_conv_{idx:06d}_{intent['name']}"

            rows = [
                {
                    "doc_id": doc_id,
                    "conv_id": conv_id,
                    "turn_id": 1,
                    "speaker": "student",
                    "text": intent["user_q"],
                },
                {
                    "doc_id": doc_id,
                    "conv_id": conv_id,
                    "turn_id": 2,
                    "speaker": "agent",
                    "text": intent["agent_a"],
                },
                {
                    "doc_id": doc_id,
                    "conv_id": conv_id,
                    "turn_id": 3,
                    "speaker": "student",
                    "text": intent["user_follow"],
                },
                {
                    "doc_id": doc_id,
                    "conv_id": conv_id,
                    "turn_id": 4,
                    "speaker": "agent",
                    "text": intent["agent_follow"],
                },
            ]

            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generated {conversations} synthetic conversations in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/synthetic_university_calls.jsonl"),
    )
    parser.add_argument("--conversations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(output=args.output, conversations=args.conversations, seed=args.seed)
