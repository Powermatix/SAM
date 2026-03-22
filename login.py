"""
Run this script once to authenticate with HuggingFace.
It will prompt you for your HF token (read-only token is fine).

Make sure you have already accepted the SAM3 license at:
  https://huggingface.co/facebook/sam3
"""
from huggingface_hub import login

print("=" * 50)
print("  HuggingFace Login for SAM3")
print("=" * 50)
print()
print("You need a HuggingFace token with read access.")
print("Get one at: https://huggingface.co/settings/tokens")
print()
login()
print()
print("Login successful! You can now run:  python test_sam3.py")
