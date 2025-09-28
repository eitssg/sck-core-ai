#!/usr/bin/env python3
"""
Test script for the SCK Core AI indexing system.

This script verifies that the indexing components work correctly
and can index documentation and codebase content.
"""

import os
import sys
from pathlib import Path
import logging

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_indexing_availability():
    """Test if indexing components are available."""
    print("Testing indexing system availability...")

    try:
        from core_ai.indexing import get_availability_status

        status = get_availability_status()
        print("\nIndexing System Status:")

        for component, info in status.items():
            status_text = "✅ Available" if info["available"] else "❌ Not Available"
            print(f"  {component}: {status_text}")
            if not info["available"] and info["error"]:
                print(f"    Error: {info['error']}")

        return all(info["available"] for info in status.values())

    except Exception as e:
        print(f"❌ Failed to check availability: {e}")
        return False


def test_vector_store():
    """Test vector store functionality."""
    print("\nTesting VectorStore...")

    try:
        from core_ai.indexing import VectorStore

        # Create a temporary vector store
        vector_store = VectorStore()

        # Test adding and searching
        test_docs = [
            "This is a test document about Python programming.",
            "Another document about CloudFormation templates.",
            "Documentation about AWS Lambda functions.",
        ]

        test_metadata = [
            {"source": "test", "type": "python"},
            {"source": "test", "type": "cloudformation"},
            {"source": "test", "type": "lambda"},
        ]

        # Add documents
        success = vector_store.add_documents(
            collection_name="documentation",
            documents=test_docs,
            metadatas=test_metadata,
        )

        if success:
            print("  ✅ Successfully added test documents")

            # Test search
            results = vector_store.search_documents(
                collection_name="documentation", query="Python programming", n_results=2
            )

            if results:
                print(f"  ✅ Search returned {len(results)} results")
                return True
            else:
                print("  ❌ Search returned no results")
                return False
        else:
            print("  ❌ Failed to add test documents")
            return False

    except Exception as e:
        print(f"  ❌ VectorStore test failed: {e}")
        return False


def test_context_manager():
    """Test context manager functionality."""
    print("\nTesting ContextManager...")

    try:
        from core_ai.indexing import ContextManager

        # Auto-detect workspace paths
        workspace_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../")
        )
        build_directory = os.path.join(workspace_root, "sck-core-docs", "build")

        print(f"  Workspace root: {workspace_root}")
        print(f"  Build directory: {build_directory}")

        # Create context manager
        context_manager = ContextManager(
            build_directory=build_directory, workspace_root=workspace_root
        )

        # Test getting system stats (should work without indexes)
        stats = context_manager.get_system_stats()

        if "documentation" in stats and "codebase" in stats:
            print("  ✅ ContextManager created successfully")
            print(
                f"  📊 Documentation chunks: {stats['documentation'].get('total_chunks', 0)}"
            )
            print(
                f"  📊 Codebase elements: {stats['codebase'].get('total_elements', 0)}"
            )
            return True
        else:
            print("  ❌ ContextManager stats incomplete")
            return False

    except Exception as e:
        print(f"  ❌ ContextManager test failed: {e}")
        return False


def test_build_documentation_exists():
    """Test if built documentation exists."""
    print("\nChecking for built documentation...")

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    build_directory = Path(workspace_root) / "sck-core-docs" / "build"

    if not build_directory.exists():
        print(f"  ❌ Build directory not found: {build_directory}")
        print("  💡 You may need to build the documentation first:")
        print("     cd sck-core-docs && .\\build.ps1 all")
        return False

    # Check for expected documentation sections
    sections = ["technical_reference", "developer_guide", "user_guide"]
    found_sections = []

    for section in sections:
        section_path = build_directory / section
        if section_path.exists():
            html_files = list(section_path.rglob("*.html"))
            if html_files:
                found_sections.append(section)
                print(f"  ✅ Found {section}: {len(html_files)} HTML files")
            else:
                print(f"  ❌ {section} directory exists but no HTML files found")
        else:
            print(f"  ❌ {section} directory not found")

    if found_sections:
        print(f"  ✅ Found {len(found_sections)} documentation sections")
        return True
    else:
        print("  ❌ No valid documentation sections found")
        return False


def test_mcp_server_creation():
    """Test MCP server creation."""
    print("\nTesting MCP Server creation...")

    try:
        from core_ai.mcp_server import SCKCoreAIMCPServer

        # Create server (this should work even without MCP dependencies for basic testing)
        server = SCKCoreAIMCPServer()

        if server:
            print("  ✅ MCP Server created successfully")

            # Check if indexing system is available
            if server.context_manager:
                print("  ✅ Indexing system integrated")
                return True
            else:
                print("  ⚠️  MCP Server created but indexing system not available")
                return True
        else:
            print("  ❌ Failed to create MCP Server")
            return False

    except Exception as e:
        print(f"  ❌ MCP Server creation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 SCK Core AI Indexing System Test Suite")
    print("=" * 50)

    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing

    tests = [
        ("Indexing Availability", test_indexing_availability),
        ("Built Documentation", test_build_documentation_exists),
        ("Vector Store", test_vector_store),
        ("Context Manager", test_context_manager),
        ("MCP Server", test_mcp_server_creation),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n🏆 {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! The indexing system is ready to use.")
        print("\nNext steps:")
        print("1. Initialize indexes: python run_mcp_server.py --initialize-indexes")
        print("2. Run MCP server: python run_mcp_server.py")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        if not any(
            name == "Built Documentation" and result for name, result in results
        ):
            print("\n💡 Missing documentation? Build it with:")
            print("   cd sck-core-docs && .\\build.ps1 all")


if __name__ == "__main__":
    main()
