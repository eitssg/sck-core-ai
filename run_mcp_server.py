#!/usr/bin/env python3
"""
CLI script to run the SCK Core AI MCP server.

Usage:
    python run_mcp_server.py [--workspace-root PATH] [--build-directory PATH]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add the current directory to the Python path so we can import core_ai
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core_ai.mcp_server import SCKCoreAIMCPServer
except ImportError as e:
    print(f"Error importing MCP server: {e}")
    print("Make sure you've installed dependencies with: uv sync")
    sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run SCK Core AI MCP server")
    parser.add_argument(
        "--workspace-root",
        type=str,
        help="Path to workspace root directory (auto-detected if not provided)",
    )
    parser.add_argument(
        "--build-directory",
        type=str,
        help="Path to documentation build directory (auto-detected if not provided)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--initialize-indexes",
        action="store_true",
        help="Initialize indexes on startup",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    try:
        # Create and start the MCP server
        logger.info("Starting SCK Core AI MCP Server...")

        server = SCKCoreAIMCPServer(workspace_root=args.workspace_root, build_directory=args.build_directory)

        # Initialize indexes if requested
        if args.initialize_indexes and server.context_manager:
            logger.info("Initializing indexes...")
            try:
                results = server.context_manager.initialize_indexes()
                logger.info(f"Index initialization completed: {results}")
            except Exception as e:
                logger.error(f"Index initialization failed: {e}")

        # Run the server
        asyncio.run(server.run())

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
