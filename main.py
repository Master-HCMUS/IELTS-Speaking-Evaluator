"""
Main entry point for the IELTS Speaking Audio Recorder.

This module provides the primary entry point for running the audio recorder
application. It can be executed directly or imported as a module.
"""

import sys
from src.cli import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)