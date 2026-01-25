#!/usr/bin/env python3
"""
Tribler Search CLI Tool

This script uses the Tribler REST API to search for torrents in the local database
and display the results in a formatted table.

Usage:
    python tribler_search.py "search query"
    python tribler_search.py "search query" --limit 10
    python tribler_search.py "search query" --port 51766
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import termios
import tty
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich import box
from rich.text import Text
from dart.tribler_rerank import encode_search_results
from dart.common import FeatureScaler

console = Console()

def getch():
    """Get a single character from stdin without echo."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)

        # Handle Ctrl+C
        if ch == '\x03':
            raise KeyboardInterrupt

        # Handle arrow keys (escape sequences)
        if ch == '\x1b':
            # Set stdin to non-blocking mode to check for more characters
            old_flags = termios.tcgetattr(fd)
            attrs = termios.tcgetattr(fd)
            attrs[6][termios.VMIN] = 0  # Minimum bytes to read
            attrs[6][termios.VTIME] = 1  # Timeout in deciseconds (0.1s)
            termios.tcsetattr(fd, termios.TCSANOW, attrs)

            # Try to read the next 2 characters (for arrow keys like [A, [B)
            next_chars = sys.stdin.read(2)

            # Restore original terminal settings
            termios.tcsetattr(fd, termios.TCSANOW, old_flags)

            if next_chars:
                ch += next_chars
            # else: just ESC key pressed
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def display_result_item(result: dict, index: int, is_selected: bool = False) -> None:
    """Display a single result with color formatting."""
    name = result.get('name', 'Unknown')
    size = format_size(result.get('size'))
    seeders = result.get('num_seeders') or 0
    leechers = result.get('num_leechers') or 0
    infohash = result.get('infohash', 'N/A')

    # Color coding for health
    if seeders >= 10:
        health_color = "green"
    elif seeders >= 1:
        health_color = "yellow"
    else:
        health_color = "red"

    # Build the display line using Text objects to avoid markup parsing issues
    prefix = "→ " if is_selected else "  "

    # First line: index and name
    line1 = Text()
    line1.append(prefix)
    line1.append(f"{index}.", style="bold white")
    line1.append(" ")
    line1.append(name, style="bold cyan" if is_selected else "")
    console.print(line1)

    # Second line: size and health
    line2 = Text()
    line2.append("   Size: ")
    line2.append(size, style="blue")
    line2.append("  |  Health: ")
    line2.append(f"S:{seeders} L:{leechers}", style=health_color)
    console.print(line2)

    # Third line: infohash
    console.print(f"   [dim]Infohash: {infohash}[/dim]\n")


def interactive_results_browser(results: list[dict], query: str) -> tuple[str, dict | None]:
    """
    Interactive results browser with arrow key navigation.
    Shows 5 results at a time with scrolling.

    Args:
        results: List of search results
        query: The search query

    Returns:
        Tuple of (action, selected_result) where action is 'select', 'new_query', or 'quit'
    """
    if not results:
        console.print("[yellow]No results found![/yellow]")
        return ('new_query', None)

    selected_idx = 0
    window_size = 5  # Number of results to show at once
    window_start = 0  # Track window position

    while True:
        # Scroll down if selected item goes below visible window
        if selected_idx >= window_start + window_size:
            window_start = selected_idx - window_size + 1

        # Scroll up if selected item goes above visible window
        if selected_idx < window_start:
            window_start = selected_idx

        # Calculate window end
        window_end = min(window_start + window_size, len(results))

        # Clear screen and display header
        console.clear()
        console.print(Panel.fit(
            f"[bold cyan]Search Results for:[/bold cyan] [white]{query}[/white]\n"
            f"[dim]Showing {window_start + 1}-{window_end} of {len(results)} results[/dim]",
            border_style="cyan",
            box=box.ROUNDED
        ))
        console.print()

        # Display only visible results
        for idx in range(window_start, window_end):
            result = results[idx]
            display_result_item(result, idx + 1, is_selected=(idx == selected_idx))

        # Display controls
        console.print(Panel(
            "[bold]Controls:[/bold] [cyan]↑/↓[/cyan] Navigate  |  "
            "[green]ENTER[/green] Select  |  "
            "[yellow]Q[/yellow] New Query  |  "
            "[red]ESC[/red] Exit",
            border_style="blue",
            box=box.ROUNDED
        ))

        # Get user input
        key = getch()

        if key == '\x1b[A':  # Up arrow
            selected_idx = max(0, selected_idx - 1)
        elif key == '\x1b[B':  # Down arrow
            selected_idx = min(len(results) - 1, selected_idx + 1)
        elif key == '\r' or key == '\n':  # Enter
            return ('select', results[selected_idx])
        elif key.lower() == 'q':  # Q for new query
            return ('new_query', None)
        elif key == '\x1b':  # ESC
            return ('quit', None)


def get_tribler_config() -> tuple[str | None, int | None]:
    """
    Read the API key and port from the Tribler configuration file.
    Checks multiple possible locations.

    Returns:
        Tuple of (api_key, port) or (None, None) if not found
    """
    # Check multiple possible Tribler configuration locations
    possible_paths = [
        Path.home() / ".Tribler" / "8.0" / "configuration.json",      # GUI version
        Path.home() / ".tribler_cli" / "configuration.json",          # CLI version
        Path.home() / ".Tribler" / "git" / "configuration.json",      # Dev version
        Path.home() / ".Tribler" / "configuration.json",              # Alternative
    ]

    for config_path in possible_paths:
        if not config_path.exists():
            continue

        try:
            with open(config_path) as f:
                config = json.load(f)
                api_config = config.get("api", {})
                api_key = api_config.get("key")
                port = api_config.get("http_port_running") or api_config.get("http_port", 51766)

                if api_key:
                    return api_key, port
        except Exception:
            continue

    return None, None


def format_size(size_bytes: int | None) -> str:
    """Format file size in human-readable format."""
    if size_bytes is None or size_bytes == 0:
        return "N/A"

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_health(seeders: int | None, leechers: int | None) -> str:
    """Format health information."""
    s = seeders if seeders is not None else 0
    l = leechers if leechers is not None else 0
    return f"S:{s} L:{l}"


def search_tribler(query: str, port: int = 51766, limit: int = 20, api_key: str | None = None) -> dict:
    """
    Search the Tribler database using the REST API.

    Args:
        query: The search query text
        port: The Tribler REST API port (default from logs: 51766)
        limit: Maximum number of results to return
        api_key: The Tribler API key for authentication

    Returns:
        Dictionary containing search results and metadata
    """
    base_url = f"http://127.0.0.1:{port}"
    endpoint = "/api/metadata/search/local"

    params = {
        "fts_text": query,
        "first": 1,
        "last": limit,
        "hide_xxx": "false",
        "include_total": "true"
    }

    url = f"{base_url}{endpoint}?{urlencode(params)}"

    # Create request with API key header if provided
    request = Request(url)
    if api_key:
        request.add_header("X-Api-Key", api_key)

    try:
        with urlopen(request, timeout=10) as response:
            data = response.read().decode('utf-8')
            return json.loads(data)
    except URLError as e:
        if hasattr(e, 'code') and e.code == 401:
            print("Error: Unauthorized. Invalid or missing API key.")
            print("The API key is stored in: ~/.Tribler/git/configuration.json")
            print("You can pass it with: --api-key YOUR_KEY")
        else:
            print(f"Error: Could not connect to Tribler API on port {port}")
            print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error making request: {e}")
        sys.exit(1)


def send_remote_search(query: str, port: int = 51766, api_key: str | None = None) -> str:
    """
    Send a remote search request to the Tribler P2P network.

    Args:
        query: The search query text
        port: The Tribler REST API port
        api_key: The Tribler API key for authentication

    Returns:
        UUID string for tracking this search request
    """
    base_url = f"http://127.0.0.1:{port}"
    endpoint = "/api/search/remote"

    params = {
        "fts_text": query,
    }

    url = f"{base_url}{endpoint}?{urlencode(params)}"

    # Create PUT request with API key header
    request = Request(url, method="PUT")
    if api_key:
        request.add_header("X-Api-Key", api_key)

    try:
        with urlopen(request, timeout=10) as response:
            data = response.read().decode('utf-8')
            result = json.loads(data)
            return result["request_uuid"]
    except URLError as e:
        print(f"Error sending remote search: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def listen_for_results(request_uuid: str, port: int = 51766, api_key: str | None = None,
                       timeout: int = 15) -> list[dict]:
    """
    Listen to the events stream and collect remote search results.

    Args:
        request_uuid: The UUID of the search request to match
        port: The Tribler REST API port
        api_key: The Tribler API key for authentication
        timeout: How many seconds to wait for results

    Returns:
        List of torrent metadata dictionaries
    """
    base_url = f"http://127.0.0.1:{port}"
    endpoint = "/api/events"
    url = f"{base_url}{endpoint}"

    request = Request(url)
    if api_key:
        request.add_header("X-Api-Key", api_key)

    results = []
    seen_infohashes = set()
    start_time = time.time()
    result_count = 0

    console.print(f"[cyan]Searching P2P network ({timeout}s)...[/cyan]")

    try:
        with urlopen(request) as response:
            current_event_type = None

            while result_count < 20:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    break

                try:
                    line = response.readline()
                    if not line:
                        break

                    line = line.decode('utf-8').strip()
                    if not line:
                        continue

                    # Parse Server-Sent Events (SSE) format
                    if line.startswith('event:'):
                        current_event_type = line[6:].strip()
                        continue
                    elif line.startswith('data:'):
                        json_str = line[5:].strip()
                        if not json_str:
                            continue
                    else:
                        json_str = line

                    event = json.loads(json_str)
                    topic = current_event_type or event.get('topic', 'unknown')

                    if topic == "remote_query_results":
                        event_uuid = event.get("uuid")
                        peer_results = event.get("results", [])
                        peer_id = event.get("peer", "unknown")

                        if event_uuid == request_uuid:
                            new_results = 0
                            for result in peer_results:
                                infohash = result.get("infohash")
                                if infohash and infohash not in seen_infohashes:
                                    results.append(result)
                                    seen_infohashes.add(infohash)
                                    new_results += 1

                            if new_results > 0:
                                result_count += new_results
                                console.print(f"[green]✓[/green] +{new_results} from peer {peer_id[:8]}... (total: {result_count})")

                except json.JSONDecodeError:
                    continue
                except Exception:
                    if results:
                        break
                    raise

    except URLError as e:
        if not results:
            console.print(f"[red]Error connecting to event stream: {e}[/red]")
            sys.exit(1)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if not results:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    console.print(f"[green]Found {len(results)} results[/green]")
    return results


def search_tribler_remote(query: str, port: int = 51766, api_key: str | None = None,
                          timeout: int = 3) -> list[dict]:
    """
    Perform a remote search on the Tribler P2P network.

    Args:
        query: The search query text
        port: The Tribler REST API port
        api_key: The Tribler API key for authentication
        timeout: How many seconds to wait for results

    Returns:
        List of torrent metadata dictionaries
    """
    request_uuid = send_remote_search(query, port, api_key)
    return listen_for_results(request_uuid, port, api_key, timeout)


def rerank_results(results: list[dict], query: str, model: np.ndarray, scaler: FeatureScaler) -> list[dict]:
    """
    Rerank search results using a trained PDGD model.

    Args:
        results: List of search result dictionaries
        query: The search query text
        model: Model weights
        scaler: FeatureScaler for normalizing features

    Returns:
        Reranked list of search results
    """
    feature_matrix = encode_search_results(results, query, scaler)
    scores = np.dot(feature_matrix, model)

    scored_results = list(zip(results, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return [result for result, _ in scored_results]


def display_results(data: dict, query: str) -> None:
    """Display search results in a formatted table."""
    results = data.get("results", [])
    total = data.get("total", len(results))

    if not results:
        print(f"\nNo results found for query: '{query}'")
        return

    print(f"\n{'='*80}")
    print(f"Search Results for: '{query}'")
    print(f"Found {total} total results, showing {len(results)}")
    print(f"{'='*80}\n")

    for idx, result in enumerate(results, 1):
        name = result.get("name", "Unknown")
        size = format_size(result.get("size"))
        infohash = result.get("infohash", "N/A")
        category = result.get("category", "N/A")
        num_seeders = result.get("num_seeders")
        num_leechers = result.get("num_leechers")
        health = format_health(num_seeders, num_leechers)
        created = result.get("created", "N/A")

        print(f"{idx}. {name}")
        print(f"   Size: {size}  |  Category: {category}  |  Health: {health}")
        print(f"   Infohash: {infohash}")
        if created != "N/A":
            print(f"   Created: {created}")
        print()


def display_remote_results(results: list[dict], query: str) -> None:
    """Display remote search results in a formatted table."""
    if not results:
        print(f"No results found for query: '{query}'")
        return

    # Sort by health (seeders)
    results.sort(key=lambda x: (x.get("num_seeders") or 0), reverse=True)

    print(f"{'='*80}")
    print(f"Remote Search Results for: '{query}'")
    print(f"Found {len(results)} unique results from the P2P network")
    print(f"{'='*80}\n")

    for idx, result in enumerate(results, 1):
        name = result.get("name", "Unknown")
        size = format_size(result.get("size"))
        infohash = result.get("infohash", "N/A")
        category = result.get("category", "N/A")
        num_seeders = result.get("num_seeders")
        num_leechers = result.get("num_leechers")
        health = format_health(num_seeders, num_leechers)
        created = result.get("created", "N/A")

        print(f"{idx}. {name}")
        print(f"   Size: {size}  |  Category: {category}  |  Health: {health}")
        print(f"   Infohash: {infohash}")
        if created != "N/A":
            print(f"   Created: {created}")
        print()


def main():
    """Main entry point for the interactive search tool."""
    parser = argparse.ArgumentParser(
        description="Interactive search for torrents in the Tribler database",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of results to return (default: 20)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=51766,
        help="Tribler REST API port (default: 51766)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="Tribler API key for authentication (auto-detected from config if not provided)"
    )

    parser.add_argument(
        "--remote",
        action="store_true",
        help="Search the P2P network instead of local database (provides more results)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="For remote search: seconds to wait for results (default: 3)"
    )

    parser.add_argument(
        "--ltr",
        action="store_true",
        help="Rerank results using trained PDGD model"
    )

    args = parser.parse_args()

    # Get API key and port from command line or auto-detect from config
    api_key = args.api_key
    port = args.port

    # If not provided via command line, try to auto-detect
    if not api_key or port == 51766:  # 51766 is the default, might not be actual
        detected_key, detected_port = get_tribler_config()

        if not api_key and detected_key:
            api_key = detected_key
            console.print(f"[dim]Auto-detected API key from configuration[/dim]",)

        if port == 51766 and detected_port:
            port = detected_port
            console.print(f"[dim]Auto-detected Tribler running on port {port}[/dim]")

    if not api_key:
        console.print("[yellow]Warning:[/yellow] No API key found. If authentication is required, provide --api-key")
        console.print("[dim]Checked locations: ~/.Tribler/8.0/, ~/.tribler_cli/, ~/.Tribler/git/[/dim]")
        console.print()
    
    model = np.load("models/pdgd_ranker.npy")
    scaler = FeatureScaler.load("tribler_data/feature_scaler.json")

    # Interactive search loop
    console.print(Panel.fit(
        "[bold cyan]Tribler Interactive Search[/bold cyan]\n"
        "[dim]An interactive torrent search interface[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    while True:
        # Get search query from user
        try:
            query = Prompt.ask("[bold cyan]Enter search query[/bold cyan]")
            if not query or query.strip() == "":
                console.print("[yellow]Please enter a valid query[/yellow]")
                continue
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        # Perform search
        if args.remote:
            # Hybrid search: local + remote
            console.print()
            with console.status("[cyan]Searching local database...[/cyan]", spinner="dots"):
                local_data = search_tribler(query, port, args.limit, api_key)
                local_results = local_data.get('results', [])
            console.print(f"[green]Found {len(local_results)} local results[/green]")

            remote_results = search_tribler_remote(query, port, api_key, args.timeout)

            # Merge and deduplicate by infohash
            seen_infohashes = {r.get('infohash') for r in local_results if r.get('infohash')}
            merged_remote = [r for r in remote_results if r.get('infohash') not in seen_infohashes]
            results = local_results + merged_remote

            if merged_remote:
                console.print(f"[green]Added {len(merged_remote)} unique remote results[/green]")
        else:
            with console.status(f"[cyan]Searching for '{query}'...[/cyan]", spinner="dots"):
                data = search_tribler(query, port, args.limit, api_key)
                results = data.get('results', [])

        if args.ltr and results:
            console.print("[cyan]Reranking results...[/cyan]")
            results = rerank_results(results, query, model, scaler)

        # Enter interactive results browser
        if results:
            try:
                action, selected_result = interactive_results_browser(results, query)

                if action == 'select':
                    console.clear()
                    selected_name = selected_result.get('name', 'Unknown')

                    # Build panel content using Text to avoid markup issues
                    panel_content = Text()
                    panel_content.append("Selected: ", style="green")
                    panel_content.append(selected_name)
                    panel_content.append("\n")
                    panel_content.append(f"Infohash: {selected_result.get('infohash', 'N/A')}", style="dim")

                    console.print(Panel(
                        panel_content,
                        border_style="green",
                        box=box.ROUNDED
                    ))
                    console.print("\n[dim]Press Enter to continue...[/dim]")
                    input()
                    # Continue to new query
                elif action == 'new_query':
                    # Continue to next iteration
                    continue
                elif action == 'quit':
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
        else:
            console.print(f"[yellow]No results found for: '{query}'[/yellow]")
            console.print("[dim]Press Enter to try a new search...[/dim]")
            input()


if __name__ == "__main__":
    main()
