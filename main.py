"""
main.py

Purpose
-------
This file is the entry point for the project. Right now it:
    1. Displays a welcome message when the application starts.
    2. Configures a basic logging setup for the project.
    3. Acts as a placeholder / foundation for future project logic
       (e.g., data loading, model training, inference, or API startup).

As the project grows, additional modules (data preprocessing, model
training, utilities, etc.) can be imported and orchestrated from here.
"""

import logging


def configure_logging() -> None:
    """
    Configure the root logger for the application.

    Sets a consistent log format and level so every module in the
    project can use `logging.getLogger(__name__)` and have its
    messages formatted the same way.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def display_welcome_message() -> None:
    """Print a simple welcome message to the console on startup."""
    print("=" * 50)
    print("  Welcome to the Project!")
    print("  This is the main entry point of the application.")
    print("=" * 50)


def main() -> None:
    """
    Orchestrate application startup.

    Currently handles logging setup and a welcome message.
    Future project logic (data pipeline, model training, etc.)
    should be added or called from here.
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    display_welcome_message()
    logger.info("Application started successfully.")

    # TODO: Add future project logic here
    # e.g., load config, initialize data pipeline, start training, etc.


if __name__ == "__main__":
    main()
